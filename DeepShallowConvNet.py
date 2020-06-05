
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


##########################################
### DEEPCONVNET #########################
# Enk DeepConvNet
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)

    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/EnK_DeepConvNet'+SaveMatFileName, patience,delta), folder+'/EnK_DeepConvNet'+SaveMatFileName, numOfEpochs,
             samplingRate, "EnK_DeepConvNet",dropoutRate,visibleGPU)

# DeepNet
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)

    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/DeepConvNet'+SaveMatFileName, patience,delta), folder+'/DeepConvNet'+SaveMatFileName, numOfEpochs,
             samplingRate, "DeepConvNet",dropoutRate,visibleGPU)

# #DeepNet with Gaussian Noise
# for i in range(start, totalDataLength, 1):
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName, dataVar, labelVar, numOfClasses, numOfKernels, scaleFactor,
#              BS, checkpointCallbacks(folder+'/Gau_DeepConvNet' + SaveMatFileName, patience,delta), folder+'/Gau_DeepConvNet' + SaveMatFileName, numOfEpochs,
#              samplingRate, "Gau_DeepConvNet", dropoutRate,visibleGPU)



####################################################################
################ SHALLOWCONTNet ####################################
# EnK
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/EnK_ShallowConvNet'+SaveMatFileName, patience,delta), folder+'/EnK_ShallowConvNet'+SaveMatFileName, numOfEpochs,
             samplingRate, "EnK_ShallowConvNet",dropoutRate,visibleGPU)

# DeepNet
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/ShallowConvNet'+SaveMatFileName, patience,delta), folder+'/ShallowConvNet'+SaveMatFileName, numOfEpochs,
             samplingRate, "ShallowConvNet",dropoutRate,visibleGPU)

# # ShallowNet with Gaussian Noise
# for i in range(start, totalDataLength, 1):
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName, dataVar, labelVar, numOfClasses, numOfKernels, scaleFactor,
#              BS, checkpointCallbacks(folder+'/Gau_ShallowConvNet' + SaveMatFileName, patience,delta), folder+'/Gau_ShallowConvNet' + SaveMatFileName, numOfEpochs,
#              samplingRate, "Gau_ShallowConvNet", dropoutRate,visibleGPU)



#####################################################################################
################ ConvGuas ##########################################################
#
# for i in range(start, totalDataLength,1):
#
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
#                  BS, checkpointCallbacks(folder+'/ConvGau_EEGNet'+SaveMatFileName, patience,delta), folder+'/ConvGau_EEGNet'+SaveMatFileName, numOfEpochs,
#              samplingRate, "ConvGau_EEGNet",dropoutRate,visibleGPU)
#
#
# # Deep
# # numOfEpochs=300
# for i in range(start, totalDataLength,1 ):
#
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
#                  BS, checkpointCallbacks(folder+'/ConvGau_DeepConvNet'+SaveMatFileName, patience,delta), folder+'/ConvGau_DeepConvNet'+SaveMatFileName, numOfEpochs,
#              samplingRate, "ConvGau_DeepConvNet",dropoutRate,visibleGPU)
#
# # EEGNt with Gaussian Noise
# # numOfEpochs=400
# for i in range(start, totalDataLength, 1):
#     LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)
#     kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
#     modelRun(Path, LoadMatFileName, dataVar, labelVar, numOfClasses, numOfKernels, scaleFactor,
#              BS, checkpointCallbacks(folder+'/ConvGau_DeepConvNet' + SaveMatFileName, patience,delta), folder+'/ConvGau_ShallowConvNet' + SaveMatFileName, numOfEpochs,
#              samplingRate, "ConvGau_ShallowConvNet",dropoutRate, visibleGPU)