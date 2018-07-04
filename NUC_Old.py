from helper import *
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

from datetime import datetime
random.seed(datetime.now())

def createResultMatrix(Q,R,B,b):
    #Q = column matrix
    #B = referenceColumn
    #R = row matrix
    #b = referenceRow
    n = len(Q)
    m = len(Q[0])
    
    E = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(0)
        E.append(row)
    
    for y in range(m-B):
        E[b][B+y] = E[b][B+y-1] + Q[b][B+y]
    for y in range(B):
        E[b][B-y] = E[b][B-y+1] + Q[b][B-y]
    for x in range(n-b):
        E[b+x][B] = E[b+x-1][B] + R[b+x][B]
    for x in range(b):
        E[b-x][B] = E[b-x+1][B] + R[b-x][B]
    
    eCopy = copy.deepcopy(E)
        
    for i in range(n):
        for j in range(m):
            if j<B and i<b:
                E[i][j] = (Q[i][j] + E[i][j+1] + R[i][j] + E[i+1][j])/2.0
            if j<B and i>b:
                E[i][j] = (Q[i][j] + E[i][j+1] + R[i][j] + E[i-1][j])/2.0
            if j>B and i<b:
                E[i][j] = (Q[i][j] + E[i][j-1] + R[i][j] + E[i+1][j])/2.0
            if j>B and i>b:
                E[i][j] = (Q[i][j] + E[i][j-1] + R[i][j] + E[i-1][j])/2.0
    return E
    
def createColumnDifferenceMatrix(primaryImage,columnShiftedImage,referenceColumn):
    Q = []
    for i in range(len(primaryImage)):
        newRow = []
        for j in range(len(primaryImage[i])):
            val = 0
            if j<referenceColumn:
                val = columnShiftedImage[i][j]-primaryImage[i][j+1]
            if j == referenceColumn:
                val = 0
            if j>referenceColumn:
                val = primaryImage[i][j] - columnShiftedImage[i][j-1]
            newRow.append(val)
        Q.append(newRow)
    return Q

def createRowDifferenceMatrix(primaryImage,rowShiftedImage,referenceRow):
    Q = []
    for i in range(len(primaryImage)):
        newRow = []
        for j in range(len(primaryImage[i])):
            val = 0
            if i<referenceRow:
                val = rowShiftedImage[i][j]-primaryImage[i+1][j]
            if i == referenceRow:
                val = 0
            if i>referenceRow:
                val = primaryImage[i][j] - rowShiftedImage[i-1][j]
            newRow.append(val)
        Q.append(newRow)
    return Q    
    
def createSourceArray(width, baseTemp = 100):
    arr = []
    midPoint = (width-1)/2
    for j in range(width):
        newColumn = []
        for i in range(width):
            dist = np.sqrt((midPoint-i)**2 + (midPoint-j)**2)
            temp = round(baseTemp-dist,3)# - (0.5 + 0.5*random.random())*
            newColumn.append(temp)
        arr.append(newColumn)
    return arr

def createDetectorArray(width,e = 0.4):
    arr = []
    for j in range(width):
        newRow = []
        for i in range(width):
            error = round(e-2*e*random.random(),3)
            newRow.append(error)
        arr.append(newRow)
    return arr

def createImage(source,detector,colOffset = 0,rowOffset = 0):
    midS = (len(source)-1)/2
    midD = (len(detector)-1)/2
    offset = (len(source)-len(detector))/2
    startI = offset + rowOffset
    startJ = offset + colOffset
    ii,jj = 0,0
    
    image = []
    while ii<len(detector):
        row = []
        i = ii + startI
        jj = 0
        while jj<len(detector[ii]):
            j = jj + startJ
            row.append( source[i][j] + detector[ii][jj])
            jj+=1
        image.append(row)
        ii+=1
    return image

def getNonUniformity(detector,B,b):
    image = []
    for i in range(len(detector)):
        row = []
        string = ""
        for j in range(len(detector[i])):
            row.append(detector[i][j] - detector[b][B])
        image.append(row)
    return image
        
def test():
    _sourceWidth = 100
    _detectorWidth = 80
    B = 12
    b = 12
    detector = createDetectorArray(_detectorWidth)
    source = createSourceArray(_sourceWidth)
    
    primaryImage = createImage(source,detector) 
    columnShiftedImage = createImage(source,detector,colOffset = 1)
    rowShiftedImage = createImage(source,detector,rowOffset = 1)

    columnDifferenceMatrix = createColumnDifferenceMatrix(primaryImage,columnShiftedImage,B)
    rowDifferenceMatrix = createRowDifferenceMatrix(primaryImage,rowShiftedImage,b)
    resultMatrix = createResultMatrix(columnDifferenceMatrix,rowDifferenceMatrix,B,b)

    NU = getNonUniformity(detector,b,B)
    
    resultMatrix = np.asarray(resultMatrix)
    NU = np.asarray(resultMatrix)
    source = np.asarray(source)
    print NU
    print resultMatrix
    
    
    
    plt.subplot(221)
    plt.imshow(NU)
    plt.subplot(222)
    plt.imshow(resultMatrix)
    plt.subplot(223)
    plt.imshow(resultMatrix-NU)
    plt.subplot(224)
    plt.imshow(source)
    plt.show()

def run(rootDir):
    B = 0#imgWidth/2
    b = 0#imgHeight/2
    
    primaryImage = getAverageImageFromFolder(os.path.join(rootDir,"Primary"),delimiter = ".csv").tolist()
    columnShiftedImage = getAverageImageFromFolder(os.path.join(rootDir,"ColumnShifted"),delimiter = ".csv").tolist()
    rowShiftedImage = getAverageImageFromFolder(os.path.join(rootDir,"RowShifted"),delimiter = ".csv").tolist()

    columnDifferenceMatrix = createColumnDifferenceMatrix(primaryImage,columnShiftedImage,B)
    rowDifferenceMatrix = createRowDifferenceMatrix(primaryImage,rowShiftedImage,b)
    resultMatrix = createResultMatrix(columnDifferenceMatrix,rowDifferenceMatrix,B,b)
    plt.subplot(221)
    plt.imshow(np.asarray(primaryImage))
    plt.subplot(222)
    plt.imshow(np.asarray(columnShiftedImage))
    plt.subplot(223)
    plt.imshow(np.asarray(rowShiftedImage))
    plt.subplot(224)
    plt.imshow(np.asarray(resultMatrix))
    plt.show()
    plt.show()
    
if __name__ == "__main__":
    #test()
    run("D:\\A320 Calibration\\NUC 3")
    run("D:\\A320 Calibration\\NUC 1")
    run("D:\\A320 Calibration\\NUC 0.1")