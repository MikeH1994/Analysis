from helper import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

def runSSE(rootDir, plot = True,conversionFn = None):
    d = dict()
    folderList = getFolderList(rootDir,".csv")
    for folder in folderList:
        filesInFolder = getFilesInFolder(folder,".csv")
        apertureSize = getFolderName(folder)
        mu,sigma = getMuSigmaFromFolder(folder,".csv")
        if not apertureSize in d:
            d[apertureSize] = []
        d[apertureSize].append(mu)
    x = []
    y = []
    sigma = []
    for key in d:
        vals = d[key]
        vals = np.asarray(vals)
        avg = np.mean(vals)
        std = np.std(vals)
        print "{}\t".format(key) + "\t".join(str(n) for n in vals)
        apertureSize = float (key.replace("mm",""))
        x.append(apertureSize)
        y.append(avg)
        sigma.append(std)
    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)
    if plot:
        plt.errorbar(x,y,yerr = sigma,fmt = 'o')
        plt.show()
if __name__ == "__main__":
    if len(sys.argv)>1:
        folder = sys.argv[1]
        print "=============================\nRunning SSE in folder {}\n=============================".format(folder)
        runSSE(folder)
    else:
        print "Usage: python <script> <directory of SSE folder>"