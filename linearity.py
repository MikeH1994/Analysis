from helper import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

def runLinearity(rootDir, plot = True,conversionFn = None,outputFilename = None):
    d = dict()
    folderList = getFolderList(rootDir)
    for integrationTimeFolder in folderList:
        integrationTime = getFolderName(integrationTimeFolder).replace("uS","")
        mu,std = getMuSigmaFromFolder(integrationTimeFolder)
        d[integrationTime] = [mu,std]
    x = []
    y = []
    sigma = []
    for key in d:
        integrationTime = float (key)
        avg,std = d[key]
        x.append(integrationTime)
        y.append(avg)
        sigma.append(std)

    data = [[x[i],y[i]] for i in range(len(x))]
    data = sorted(data,key=lambda l:l[0])
    x = [data[i][0] for i in range(len(x))]
    y = [data[i][1] for i in range(len(x))]
    if plot:
        plt.errorbar(x,y,yerr = sigma,fmt = 'o')
        plt.plot(x,y)
        plt.show()
    if outputFilename:
        outputFilename = ""
        
if __name__ == "__main__":
    if len(sys.argv)>1:
        folder = sys.argv[1]
        print "=============================\nRunning Linearity in folder {}\n=============================".format(folder)
        runLinearity(folder)
    else:
        #print "Usage: python <script> <root directory of SSE folder>"
        runLinearity("D:\\CEDIP Calibration\\300degC\\Linearity")