
'''
EnK Layer
© Avinash K Singh 
https://github.com/thinknew/enk
Licensed under MIT License
'''


import scipy.io as sio
import tensorflow.compat.v1 as tf
from otherClassifier import *
from models import EEGNet, Gau_EEGNet, ShallowConvNet, Gau_ShallowConvNet, DeepConvNet, Gau_DeepConvNet,\
    ConvGau_DeepConvNet, ConvGau_EEGNet, ConvGau_ShallowConvNet,CRNN
from Old_Code.ShallowConvNetModel import ShallowConvNet
import time
import os
from op import *
import cv2
from GradCAM.grad_cam import *

def modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
             BS, checkpoint, SaveMatFileName, numOfEpochs, samplingRate,type,dropoutRate, visibleGPU):

    os.environ["CUDA_VISIBLE_DEVICES"] = visibleGPU
    tf.keras.backend.clear_session()

    with tf.Graph().as_default() as g:
    # Load data file
        mat = sio.loadmat(Path + LoadMatFileName)
        labels= (mat[labelVar]) # Labels
        data=mat[dataVar] # Data

        # Variables Initialization
        numOfSamples=data.shape[1]
        numOfChannels=data.shape[0]
        numOfTrials = data.shape[2]

        kernelLength = (int)(samplingRate / 2)  #
        EEGNet_F1 = (int)(2 * numOfChannels)  # Double to the number of Channels
        EEGNet_F2 = (int)(4 * EEGNet_F1)  # Double to the EEGNet_F1

        if type=="EnK_EEGNet":
            model = EEGNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                           dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=True,
                           dropoutType='Dropout')
        elif type=="EEGNet":
            model = EEGNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                           dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                           dropoutType='Dropout')
        elif type == "Gau_EEGNet":
            model = Gau_EEGNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                           dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                           dropoutType='Dropout')
        elif type == "EnK_ShallowConvNet":
            model = ShallowConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                           dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=True,
                           dropoutType='Dropout')
        elif type == "ShallowConvNet":
            model = ShallowConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "Gau_ShallowConvNet":
            model = Gau_ShallowConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "EnK_DeepConvNet":
            model = DeepConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=True,
                                   dropoutType='Dropout')
        elif type == "DeepConvNet":
            model = DeepConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "Gau_DeepConvNet":
            model = Gau_DeepConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "ConvGau_DeepConvNet":
            model = ConvGau_DeepConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "ConvGau_ShallowConvNet":
            model = ConvGau_ShallowConvNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        elif type == "ConvGau_EEGNet":
            model = ConvGau_EEGNet(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')

        elif type == "CRNN":
            model = CRNN(nb_classes=numOfClasses, Chans=numOfChannels, Samples=numOfSamples,
                                   dropoutRate=dropoutRate, kernLength=kernelLength, F1=EEGNet_F1, D=2, F2=EEGNet_F2, EnK=False,
                                   dropoutType='Dropout')
        else:
            print("Error: no such model exist")



        # Selecting the class
        f1_avg,pos_label, loss_type, class_weights =getClassInfo(numOfClasses)

        # extract raw data. scale by scaleFactor due to scaling sensitivity in deep learning
        X = data*scaleFactor

        print(X.shape)

        X=np.reshape(X,(numOfTrials,numOfChannels,numOfSamples)) # format should be in (trials, channels, samples)
        y = np.asarray(labels)
        Y= y.reshape(-1)

        # convert data to NCHW (trials, kernels, channels, samples) format. Data
        X      = X.reshape(numOfTrials,numOfKernels,numOfChannels, numOfSamples)

        print (model.summary())

        model.compile(loss=loss_type, optimizer='adam',metrics=['accuracy'])

        seed=7 # Fix number

        CNNacc=[]
        CNNmse=[]
        CNNf1=[]
        CNNct=[]

        X_train,X_test,Y_train, Y_test= getTrainTestVal(X, Y, testSize=0.2)

        Y_train = oneHot(Y_train, numOfClasses,True)

        Y_test = oneHot(Y_test, numOfClasses,True)

        # fitting existing model
        # model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
        #                         batch_size=BS, epochs=numOfEpochs, verbose=2, class_weight=class_weights,
        #                         callbacks=checkpoint)

        # model.fit(X_train, Y_train,
        #           batch_size=BS, epochs=numOfEpochs, verbose=2, class_weight=class_weights,
        #           callbacks=checkpoint,  validation_split = 0.20)

        model.fit(X_train, Y_train,
                  batch_size=BS, epochs=numOfEpochs, verbose=2, class_weight=class_weights,
                  callbacks=checkpoint)

        tic = time.clock()
        predicted = model.predict(X_test)
        toc = time.clock()
        # predicted= predicted
        computation_time = toc - tic


        predicted =oneHot(predicted.argmax(axis=-1), numOfClasses,False)
        mse,mae,co_kap_sco,acc,avg_pre_sco,precision,recall,\
        f1_sc=getPerformanceMetricsDL(numOfClasses, pos_label, f1_avg, Y_test, predicted)

        # Grad-Cam method with Test Data
        predicted_class = predicted.argmax(axis=-1)
        camTest, heatmapTest = grad_cam(model, X_test[2,:,:,:].reshape(1,1,numOfChannels,numOfSamples), predicted_class[2], "en_k_layer",numOfClasses)
        cv2.imwrite(SaveMatFileName+'Test.png', camTest)

        # Grad-Cam method with Test Data
        predicted_class = Y_train.argmax(axis=-1)
        camTrain, heatmapTrain = grad_cam(model, X_train[2, :, :, :].reshape(1, 1, numOfChannels, numOfSamples), predicted_class[2],
                                "en_k_layer", numOfClasses)
        cv2.imwrite(SaveMatFileName + 'Train.png', camTrain)

        print('acc, f1 score, coh kappa is ', acc, ' ', f1_sc, ' ', co_kap_sco)


        #########################################################
        # For classifiers
        # reshape back to (trials, channels, samples)
        X = X.reshape(numOfTrials, numOfChannels,numOfSamples)

        # convert labels to one-hot encodings.
        Y =oneHot(Y, numOfClasses,True)

        # Disabled the running basic classifiers
        # other_acc, other_mse, other_mae, other_avpc, \
        # other_cks, other_pre, other_rec, other_f1, other_ct = Classifiers(X, Y.argmax(axis=-1),f1_avg,numOfClasses)

        other_acc, other_mse, other_mae, other_avpc, \
        other_cks, other_pre, other_rec, other_f1, other_ct = [0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]

        other_acc.append(acc)
        other_mse.append(mse)
        other_mae.append(mse)
        other_avpc.append(avg_pre_sco)
        other_cks.append(co_kap_sco)
        other_pre.append(precision)
        other_rec.append(recall)
        other_f1.append(f1_sc)
        other_ct.append(computation_time)

        print("Classifier ACC for LogRef, LDA, L-SVM, RBF-SVM, NN, Proposed:", other_acc)
        print("Classifier MSE for LogRef, LDA, L-SVM, RBF-SVM, NN, Proposed :", other_mse)
        print("Classifier f1 score for LogRef, LDA, L-SVM, RBF-SVM, NN, Proposed :", other_f1)


        sio.savemat(SaveMatFileName+'.mat', {"acc": other_acc, "mse": other_mse,"mae": other_mae,"avg_pre_recl": other_avpc,
            "cohen_kappa": other_cks,"precision": other_pre,"recall": other_rec,"f1": other_f1, "times_prediction": other_ct,
                                             "pre_labels":predicted.argmax(axis=-1),"true_labels":Y_test.argmax(axis=-1),
                                             "camTest":camTest,"camheatmapTest":heatmapTest,
                                             "camTrain":camTrain,"camheatmapTrain":heatmapTrain,
                                             "camData":X_train[2, :, :, :].reshape(1, 1, numOfChannels, numOfSamples),
                                             "camLabel":predicted_class})


