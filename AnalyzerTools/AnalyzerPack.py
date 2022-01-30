import pandas as pd
import numpy as np
import keras
import os
import plotly_express as px
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #Supresses some unnecessary logs
mainDirectory = os.path.dirname(os.path.abspath(__file__))
mainDirectory = mainDirectory[0:-14]


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

def formatPredictions(listOfSingleElementArrays):
    i = 0
    for array in listOfSingleElementArrays:
        listOfSingleElementArrays[i] = array[0]
        i += 1
    listOfElements = [float(item) for item in listOfSingleElementArrays]
    return listOfElements

def modelPredict(modelPath, X_test):
        modelPath = modelPath.replace("\\", "/")
        model = keras.models.load_model(modelPath)
        prediction = model.predict(X_test)
        modelPathName = os.path.split(modelPath)
        modelName = modelPathName[1]
        print("\nPredicted:" + modelName + "\n")
        return list(prediction[:,0])

def modelPredictionStats(y_test, prediction):
    """
    Mean squared error, R2, correlation, stdev y_test, stdev prediciton.
    """
    correlation = np.cov(y_test, prediction)[0,1] / (np.std(y_test) * np.std(prediction))
    return mean_squared_error(y_test, prediction), r2_score(y_test, prediction), correlation, np.std(y_test), np.std(prediction)


def analyzeModel(datasetFileName, validatioDataStartingDate):
    """
    Draws a graph of the predictions and true values for the validation period. Opens in browser. Use the same validatioDataStartingDate and dataset that was used for training.
    This function uses the same openDataset function as training to separate the validation data in same fashion.
    """
    X_train, y_train, X_test, y_test, dates = openDataset(datasetFileName, validatioDataStartingDate)
    modelPath = input("ModelPath? ")
    prediction = modelPredict(modelPath, X_test)
    modelPath = modelPath.replace("\\", "/")
    modelName = os.path.split(modelPath)[1]
    stats = modelPredictionStats(y_test, prediction)
    print(stats)

    df = pd.DataFrame({
        "Predicion":prediction,
        "Reality": y_test
    },index= dates)

    fig = px.line(
        data_frame=df,
        title=modelName,
        labels={"value": "Change in S&P500" + " (%)",
                "Date": "Date",
                "variable": ""
        }
        )
    fig.show()

def multiModelAnalyzer(datasetFileName, validatioDataStartingDate):
    """
    Makes an excel spreadsheet consisting of values Model name, Mean squared error, "R2", correlation, stdev. target, stdev. prediciton for each model. Use the same validatioDataStartingDate and dataset that was used for training.
    This function uses the same openDataset function as training to separate the validation data in same fashion.
    """
    modelsFolder= input("\nModelsFolder? ")
    modelsFolder = modelsFolder.replace("\\", "/")
    modelList = os.listdir(modelsFolder)
    X_train, y_train, X_test, y_test, dates = openDataset(datasetFileName, validatioDataStartingDate)
    modelsDataFrame = pd.DataFrame(columns=["Model name", "Mean squared error", "R2", "correlation", "stdev. reality", "stdev. prediciton"])

    for model in modelList:
        prediction = modelPredict(f"{modelsFolder}/{model}", X_test)
        modelStats = modelPredictionStats(y_test, prediction)
        modelStats = [model, *modelStats]
        modelsDataFrame.loc[len(modelsDataFrame)] = modelStats

    modelsDataFrame.to_excel(f"{mainDirectory}/Data/Out/Analyzes/ModelTables/{os.path.basename(modelsFolder)}.xlsx")
