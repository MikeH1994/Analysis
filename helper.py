import cv2
import numpy as np
import os
import tkFileDialog
import warnings
from Tkinter import *
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import quad
from math import hypot
import copy

_C1_ = 1.19E-16
_C2_ = 0.014388

def temperatureToSpectralRadiance(T,wavelength):
    return _C1_*np.power(wavelength,-5) /(np.exp(_C2_/wavelength/T)-1)

def reject_outliers(data, m=1.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    
def getAverageImageFromFolder(folderpath,extension = ".txt"):
    fileList = getFilesInFolder(folderpath,extension = extension)
    img = np.zeros(createImageFromTxt(fileList[0]).shape).astype(np.float32)
    for filepath in fileList:
        img+=createImageFromTxt(filepath)
    img/=len(fileList)
    return img
    
def getMuSigmaFromFolder(folder,extension):
    vals = []
    filesInFolder = getFilesInFolder(folder,extension)
    for filepath in filesInFolder:
        img = createImageFromTxt(filepath)
        vals.append(getMeanFromImg(img))
    vals = np.asarray(vals)
    mu = np.mean(vals)
    std = np.std(vals)
    return mu,std

def getAperture(img):
    indices = np.where(img>0)
    xmin = np.min(indices[0])
    xmax = np.max(indices[0])
    ymin = np.min(indices[1])
    ymax = np.max(indices[1])
    dx = (xmax-xmin)/2
    dy = (ymax-ymin)/2
    return (dx+dy)/2


    
def getMeanFromImg(img):
    try:
        h,w = img.shape
        x,y,r = getAperturePosAndRadius(img)
        mean = 0
        if r>6:
            mask = img[y-1:y+1,x-1:x+1]
            mean = np.mean(mask)
        else:
            mean = np.max(img)
        return mean
    except:
        return np.max(img)
    
def getAperturePosAndRadius(img,plot = False):
    img8 = convertTo8Bit(img)
    circles = cv2.HoughCircles(img8,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    if plot:
        cimg = copy.copy(img8)
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],255,2)
            cv2.circle(cimg,(i[0],i[1]),2,0,3)
        
        plt.imshow(cimg)
        plt.show()

    circle = circles[0,:][0]
    x,y,r = circle[0],circle[1],circle[2]
    return x,y,r
    
def createCircularMask(x,y,h,w,r):
    mask = np.zeros((h,w),dtype = np.float32)
    for i in range(w):
        for j in range(h):
            if (i-x)**2+(j-y)**2 < r**2:
                mask[j,i] = 1
    return mask
def createImageFromTxt(filepath,imgWidth = 320,imgHeight = 256):
    f = open(filepath)
    str = f.read()
    f.close()
    data = None
    if "Image  Data" in str:
        #cedip file
        data = str.split("Image  Data :  (DL)\n")[1]
        data = np.asarray(data.split()).reshape((imgHeight,imgWidth)).astype(np.uint16)
        return data
    if "," in str:
        return np.genfromtxt(filepath,delimiter = ",")
    if ";" in str:
        return np.genfromtxt(filepath,delimiter = ";")
    else:
        return np.genfromtxt(filepath)
        
def imgRead(imgPath):
    img = cv2.imread(imgPath,-1)
    thresh = getOTSUThreshold(img)
    thresh,img = cv2.threshold(img,thresh,0,cv2.THRESH_TOZERO)
    return img

def convertTo8Bit(img,imgMin=None,imgMax = None):
    img64 = img.astype(np.float64)
    
    if imgMin == None:
        imgMin = getNonzeroMin(img64)
    if imgMax == None:
        imgMax = getNonzeroMax(img64)

    img64 = (img64-imgMin)/(imgMax-imgMin)
    img64 = img64.clip(min=0)
    img64 = 255 * img64
    img8 = img64.astype(np.uint8)
    return img8
    
def convertImgsTo8Bit(img1,img2):
    imgMin = min(getNonzeroMin(img1),getNonzeroMin(img2))
    imgMax = max(getNonzeroMax(img1),getNonzeroMax(img2))
    img1 = convertTo8Bit(img1,imgMin = imgMin,imgMax = imgMax)
    img2 = convertTo8Bit(img2,imgMin = imgMin,imgMax = imgMax)
    return img1,img2
    
def getNonzeroMin(img):
    return np.min(img[np.nonzero(img)])

def getNonzeroMax(img):
    return np.max(img[np.nonzero(img)])
    
def getFolderName(folderpath):
    if "\\" in folderpath:
        return folderpath.split("\\")[-1]
    else:
        return folderpath.split("/")[-1]
    
def getFolderList(a_dir,extension):
    directories = []
    for path,subdirectories,files in os.walk(a_dir):
        for subdir in subdirectories:
            subpath = os.path.join(path, subdir)
            contents = os.listdir(subpath)
            imageList = [os.path.join(subpath, i) for i in contents if extension in i]
            if len(imageList)>0:
                directories.append(subpath)
    return sorted(directories, key=str.lower)

def getFilesInFolder(folderpath,extension = ".txt"):
    contents = os.listdir(folderpath)
    files =  [os.path.join(folderpath, i) for i in contents if extension in i]
    return sorted(files, key=str.lower)
    
def getOTSUThreshold(img,minVal = None,maxVal=None,nBins = 512):
    #shamelessly c&p'd from https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    #find normalized_histogram, and its cumulative distribution function
    if minVal==None:
        minVal = np.min(img)
    if maxVal==None:
        maxVal = np.max(img)
    
    
    hist = cv2.calcHist([img],[0],None,[nBins],[minVal,maxVal])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(nBins)
    fn_min = np.inf
    thresh = -1
    for i in xrange(1,nBins):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    thresh = minVal + thresh*(maxVal-minVal)/nBins
    return thresh
