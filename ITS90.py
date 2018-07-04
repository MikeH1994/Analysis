from helper import *
import os

def processITS90TempFolder(folder,extension):
    d = dict()
    folderlist = getFolderList(folder,extension = extension)
    temperature = float (getFolderName(folder).replace("degC",""))
    for integrationTimeFolder in folderlist:
        if not "uS" in integrationTimeFolder:
            continue
        #mean signal at this integration time
        integrationTime = float (getFolderName(integrationTimeFolder).replace("uS",""))
        mean,sigma = getMuSigmaFromFolder(integrationTimeFolder,extension)
        d[integrationTime] = [temperature,mean,sigma]
    return d
        
def runITS90(calibrationFolder,extension):
    data = dict()
    for path,subdirectories,files in os.walk(calibrationFolder):
        for subdir in subdirectories:
            if not "degC" in subdir:
                continue
                
            subdirectory = os.path.join(path, subdir)
            setpoint = processITS90TempFolder(subdirectory,extension)
            for integrationTime in setpoint:
                if not integrationTime in data:
                    data[integrationTime] = []
                data[integrationTime].append(setpoint[integrationTime])
    for integrationTime in data:
        arr = data[integrationTime]
        temperatures = np.asarray(data[integrationTime])[:,0]
        signal = np.asarray(data[integrationTime])[:,1]
        sigma = np.asarray(data[integrationTime])[:,2]
        if len(arr)>1:
            plt.plot(temperatures,signal,label = str (integrationTime))
            plt.scatter(temperatures,signal)
        print("Integration time  {}uS\n===============\nT(degC)\tSignal\n===============".format(integrationTime))
        for i in range(len(temperatures)):
            print "{}\t{}".format(temperatures[i],signal[i],sigma[i])
    plt.legend(loc = 1)
    plt.show()
        
if __name__ == "__main__":
    if len(sys.argv)>1:
        folder = sys.argv[1]
        print "=============================\nRunning ITS90 in folder {}\n=============================".format(folder)
        runITS90(folder)
    else:
        runITS90("D:\\A320 Calibration\\ITS90",".csv")