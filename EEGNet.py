
##############################################
### Using Basic EEGnet Model ################
#############################################

import numpy as np
import scipy.io as sio

# EEGNet-specific imports
from EEGModels import EEGNet, EEGNet_TF
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping
from modelRun import *
from op import getInputDataInfo

import tensorflow.compat.v1 as tf

start=0
totalDataLength=4
numOfEpochs=0
scaleFactor = 1000  # Fix parameter
numOfKernels = 1  # Fix parameter
visibleGPU="0"
patience=10
delta=0
dropoutRate=0.5
folder='tmp'

# # Enk EEGNet
# for i in range(start, totalDataLength,1 ):
#
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
#                  BS, checkpointCallbacks(folder+'/EnK_EEGNet'+SaveMatFileName, patience,delta), folder+'/EnK_EEGNet'+SaveMatFileName, numOfEpochs,
#              samplingRate, "EnK_EEGNet",dropoutRate,visibleGPU)
#
#
# # EEGNet
# for i in range(start, totalDataLength,1 ):
#
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
#                  BS, checkpointCallbacks(folder+'/EEGNet'+SaveMatFileName, patience,delta), folder+'/EEGNet'+SaveMatFileName, numOfEpochs,
#              samplingRate, "EEGNet",dropoutRate,visibleGPU)

# EEGNt with Gaussian Noise
for i in range(start, totalDataLength, 1):
    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName, dataVar, labelVar, numOfClasses, numOfKernels, scaleFactor,
             BS, checkpointCallbacks(folder+'/Gau_EEGNet' + SaveMatFileName, patience,delta), folder+'/Gau_EEGNet' + SaveMatFileName, numOfEpochs,
             samplingRate, "Gau_EEGNet", dropoutRate,visibleGPU)


