"""
Training model base and support functions.
"""

import numpy as np
import pandas as pd
import os
import datetime as dt
import scipy.optimize as optimize
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

mainDirectory = os.path.dirname(os.path.realpath(__file__))
mainDirectory = mainDirectory[0:-9]


def openDataset(dataset, validationDataStartingDate):
    """
    Opens and prepares the dataset for training. Splits it into training and validation data based on the date
    """
    dataset = pd.read_csv("{}/Data/In/{}".format(mainDirectory, dataset))
    validationStartIndex = dataset[dataset['Date'] == validationDataStartingDate].index[0]
    validationDataLen = len(dataset) - validationStartIndex
    dataset = dataset.set_index("Date")
    dataset = dataset*100               #Scaling the values for training and viewing purpoises. Completely arbitrary
    dataset = dataset.dropna()
    columns = list(dataset.columns)
    
    
    X_data, y_data = dataset[columns[1:]], dataset[columns[0]]
    dates = y_data.index[-validationDataLen:]

    # The following code block includes a lot of data manipulation that is difficult to explain in short. Here is link to sentdex' video explaining the process
    # https://www.youtube.com/watch?v=j-3vuBynnOE&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2
    # The part talking about the reshaping starts at 13:30

    X_train, y_train = X_data[ :-validationDataLen],(y_data[ :-validationDataLen])
    X_train, y_train = np.array(X_train), np.array(y_train)         
    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
    # Same trick for the validation data
    X_test, y_test = X_data[-validationDataLen:], (y_data[-validationDataLen:])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, y_test, dates






def startTrainingModels(trainingSetLabel, dataset, activations, losses, epochsList, numberOfLSTMLayers, denseLayer, layerSizes, batchSizesList, validationDataStartingDate, earlyStop, patience):

    newpath = r"{}/Data/Models/{}".format(mainDirectory, trainingSetLabel)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    X_train, y_train, X_test, y_test, dates = openDataset(dataset, validationDataStartingDate)
    model_number = 0
    number_of_models = len(batchSizesList)*len(numberOfLSTMLayers)*len(layerSizes)*len(epochsList)*len(losses)*len(activations)


    for epochs in epochsList:
        for layerSize in layerSizes:
            for LSTMLayer in numberOfLSTMLayers:
                for batchSize in batchSizesList:
                    for activation in activations:
                        for loss in losses:
                            modelName = "{}-model_N-{}-{}-epochs-{}-loss-{}-batch_size-{}-activaion-{}-lstm_layers-{}-dense_layer-{}-neurons-{}-validation_start".format(model_number, trainingSetLabel, epochs,loss, batchSize, activation, LSTMLayer, denseLayer, layerSize, validationDataStartingDate)
                            model_number += 1

                            tensorBoard = tf.keras.callbacks.TensorBoard(
                                log_dir=f'{mainDirectory}/Data/Logs/{trainingSetLabel}/{modelName}', histogram_freq=0, write_graph=True,
                                write_images=False, update_freq='epoch', profile_batch=2,
                                embeddings_freq=0, embeddings_metadata=None
                            )
                            if earlyStop:
                                earlyStop = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss', min_delta=0, patience=patience, verbose=0,
                                    mode='auto', baseline=None, restore_best_weights=True
                                )



                            model = Sequential()
                            for i in range(LSTMLayer-1):
                                model.add(LSTM(layerSize, return_sequences= True, activation = activation, input_shape= (X_train.shape[1], 1)))
                            model.add(LSTM(layerSize, return_sequences= False, activation = activation, input_shape= (X_train.shape[1], 1)))
                            if denseLayer:
                                model.add(Dense(1))
                            model.compile(optimizer= "adam",loss= loss)
                            print("Model {}/{}".format(model_number, number_of_models))
                            print(modelName)
                            model.fit(X_train, y_train, batch_size= batchSize, epochs= epochs, callbacks= [tensorBoard, earlyStop], shuffle= True, validation_data=(X_test, y_test))
                            model.save("{}/Data/Models/{}/{}{}".format(mainDirectory, trainingSetLabel, modelName, '.h5'))