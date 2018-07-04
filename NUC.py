from helper import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random
import copy
    
    
def createNonUniformityMatrix(A,B,C):
    #A: primary image
    #B: column (x) shifted 
    #C: row (y) shifted
    """
    Working off the basis that the top left is taken to be the reference pixel, and that B and C are moving in the left and upwards
    direcitons respectively
    """
    
    height,width = A.shape
    delta = np.zeros((height,width)).astype(np.float64)
    y = 0
    for x in range(1,width):
        delta[y][x] = B[y][x] - A[y][x-1] + delta[y][x-1]
    for x in range(0,width):
        for y in range(1,height):
            delta[y][x] = C[y][x] - A[y-1][x] + delta[y-1][x]
    
    return delta
    
def createSourceArray(width, baseTemp = 100):
    arr = np.zeros((width,width)).astype(np.float64)
    midPoint = (width-1)/2
    for i in range(width):
        for j in range(width):
            dist = np.sqrt((midPoint-i)**2 + (midPoint-j)**2)
            temp = round(baseTemp-dist,3)
            arr[i][j] = temp
    return arr

def createDetectorArray(width,e = 0.4):
    arr = np.zeros((width,width)).astype(np.float64)
    for i in range(width):
        for j in range(width):
            error = round(e-2*e*random.random(),3)
            arr[i][j] = error
    return arr
  
matrix_test = None
for image in os.listdir('path_to_dir'):
    imgraw = cv.imread(os.path.join('path_to_dir', image), 0)
    imgvector = imgraw.reshape(128*128)
    try:
        matrix_test = np.vstack((matrix_test, imgvector))
    except:
        matrix_test = imgvector

# PCA
mean, eigenvectors = cv.PCACompute(matrix_test, np.mean(matrix_test, axis=0))  
  

def runNUC(A_path,B_path,C_path,title):
    A = getAverageImageFromFolder(A_path,extension = ".csv")
    B = getAverageImageFromFolder(B_path,extension = ".csv")#ColumnShifted
    C = getAverageImageFromFolder(C_path,extension = ".csv")#RowShifted
    nonUniformity = createNonUniformityMatrix(A,B,C)
    """
    blackBody = A-nonUniformity
    plt.subplot(221)
    plt.imshow(A)
    plt.subplot(222)
    plt.imshow(B)
    plt.subplot(223)
    plt.imshow(nonUniformity)
    plt.subplot(224)
    plt.imshow(blackBody)
    plt.gcf().canvas.set_window_title(title)
    plt.show()    
    """
    return np.std(nonUniformity),np.min(nonUniformity),np.max(nonUniformity)
    
    
def runSSTH(A_path,B_path,C_path):    
    A_raw = getAverageImageFromFolder(A_path,extension = ".csv")
    B_raw = getAverageImageFromFolder(B_path,extension = ".csv")#ColumnShifted
    C_raw = getAverageImageFromFolder(C_path,extension = ".csv")#RowShifted
    h,w = A_raw.shape
    A =np.zeros(A_raw.shape)
    A[0:h,1:w] = A_raw[0:h,0:w-1]
    B = B_raw
    C = np.zeros(A_raw.shape)
    C[0:h,0:w-1] = C_raw[0:h,1:w]
    
    S = A+B+C
    SBar = np.zeros(A_raw.shape)
    SBar[A.nonzero()]+=1
    SBar[B.nonzero()]+=1
    SBar[C.nonzero()]+=1
    nonUniformity=A_raw - S/SBar
    
    
    blackBody = A_raw+nonUniformity
    plt.subplot(221)
    plt.imshow(A_raw)
    plt.subplot(223)
    plt.imshow(nonUniformity)
    plt.subplot(224)
    plt.imshow(blackBody)
    plt.show()    
    
def runExploratoryNUC():
    path = "D:\\A320 Calibration\\MiniNUC3\\"
    mid = 7
    A_path = os.path.join(path,"Primary")
    B_path = os.path.join(path,"Primary")
    C_path = os.path.join(path,"Primary")
    runSSTH(A_path,B_path,C_path)

def testNUC(imageWidth = 100):
    B = 0
    b = 0
    detector = createDetectorArray(imageWidth)
    source = createSourceArray(2*imageWidth)
    
    A = createImage(source,detector) 
    B = createImage(source,detector,colOffset = -1)
    C = createImage(source,detector,rowOffset = -1)
    NU = createNonUniformityMatrix(A,B,C)

    print resultMatrix
    print NU
    
    plt.subplot(221)
    plt.imshow(detector)
    plt.subplot(222)
    plt.imshow(source)
    plt.subplot(224)
    plt.imshow(NU)
    plt.show()
    
    
if __name__ == "__main__":
    if len(sys.argv)>1:
        print "=============================\nRunning SSE in folder {}\n=============================".format(sys.argv[1])
        runNUC(sys.argv[1])
    else:
        #print "Usage: python <script> <root directory of NUC folders>"
        runExploratoryNUC()

        #testNUC()#