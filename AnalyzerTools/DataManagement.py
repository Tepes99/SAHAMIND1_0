"""
Here you find all the data management functions. Only make changes if you know what you are doing
"""
import numpy as np
import pandas as pd
import os
import io
import requests
import datetime as dt
import pandas_datareader as web
import scipy.optimize as optimize


"""
Note that paths are harcoded relative to the main working directory.

Changes in hardcoded folder names break the functions as paths do not work after that
"""

mainDirectory = os.path.dirname(os.path.abspath(__file__))
mainDirectory = mainDirectory[0:-14]




def dowloadPriceData(
    fromDate,
    toDate,
    assets
    ):
    """
    Downloads desired price data and saves it in a csv file

    .../Data/Storage/Raw/AssetPrices/{asset}.csv

    Note that this overwrites data with same name in same folder

    """
    i = 1
    fromDate = dt.datetime(*fromDate)
    toDate = dt.datetime(*toDate)
    for asset in assets:
        data = web.DataReader(asset,"yahoo", fromDate, toDate)
        data = data["Adj Close"]
        data = data[~np.isnan(data)]
        data.to_csv("{}/Data/Storage/Raw/AssetPrices/{}.csv".format(mainDirectory, asset))
        print(f"{str(round(((i/len(assets))*100),2))} % of price data downloaded")
        i += 1


def returns(days, fromFile, toFile):
    """
    Calculates returns for 'days' time lag from price data file to returns data file

    from .../Data/Storage/Raw/AssetPrices/{fromFile}

    to .../Data/Storage/Raw/Y/{toFile}

    Remember to use the file extension .csv in the names for this to work
    """
    df = pd.read_csv("{}/Data/Storage/Raw/AssetPrices/{}".format(mainDirectory, fromFile), index_col="Date")

    returnsForTimeFrame = 0
    returns = []
    while len(returns) < len(df) -days:
        returnsForTimeFrame = df.iloc[len(returns) + days, 0]/df.iloc[len(returns), 0] -1
        returns.append(returnsForTimeFrame)

    df = df[:-days]
    df.insert(1, f"returns-{days}", returns)
    del df["Adj Close"]
    df.to_csv(("{}/Data/Storage/Raw/Y/{}".format(mainDirectory, toFile)))

###
#Quadratic Momentum Algo
###
def quadratic(x, a, b, c):
    """
    Quadratic function
    """
    return a*x**2+b*x+c

def secondDerivative(x, a, b):
    """
    SecondDerivative of the quadratic function
    """
    return 2*a*x+b


def quadraticMomentum(dataframe, targetColumn, nPoints):
    """
    Quadratic momentum algorithm

    dataframe is the price data to be used for the momentum data production.

    targetColumn is the column of the price data.

    nPoints is the lenght of time series data that is used for each point in time.

    For example in daily data, 21 points would look back 1-month as month has 21 tradingdays.
    """
    slopeX = np.arange(0.0, nPoints, 0.01)
    quadraticX = np.arange(nPoints)
    array = list(dataframe[targetColumn])
    newArray = dataframe[targetColumn]
    i = nPoints
    while i < len(dataframe):
        periodData = array[i-nPoints:i]
        periodData = periodData/np.average(periodData)
        popt, popc = optimize.curve_fit(quadratic, quadraticX, periodData)
        newArray[i] = secondDerivative(slopeX[-1], popt[0],popt[1])
        i += 1
    data = newArray[nPoints:]
    return data

def runQM(momentumDataFolder):
    """
    Calculates the Quatratic momentum values for a list of price data files in chosen spesific folder and saves them in another with same name
    
    Price data is taken from .../Data/Storage/Raw/AssetPrices/

    Momentum data is saved to .../Data/Storage/Raw/Momentum/{momentumDataFolder}
    """
    if os.path.isdir("{}/Data/Storage/Raw/Momentum/{}".format(mainDirectory, momentumDataFolder)) == False:
        os.mkdir("{}/Data/Storage/Raw/Momentum/{}".format(mainDirectory, momentumDataFolder))
    fileList = os.listdir("{}/Data/Storage/Raw/AssetPrices".format(mainDirectory))
    i = 1
    for filename in fileList:
        fileCount = len(fileList)
        df = pd.read_csv("{}/Data/Storage/Raw/AssetPrices/{}".format(mainDirectory, filename), index_col="Date")
        dataset = quadraticMomentum(df, "Adj Close", 21)
        dataset.to_csv("{}/Data/Storage/Raw/Momentum/{}/{}".format(mainDirectory, momentumDataFolder,filename))
        print("Running QMA " + str(round(((i/fileCount)*100),2)) + "%")
        i += 1
###
#Dataset forming from explaining X variables and explained Y variable
###
def makeDataset(
    nameForDataset,
    yFilename, 
    momentumFiles,
    momentumFolder
    ):
    """
    Makes trainable dataset from the cleaned and processed data
    """
    df = pd.read_csv(f"{mainDirectory}/Data/Storage/Raw/Momentum/{momentumFolder}/{yFilename}", index_col="Date")
    del df['Adj Close']
    for fileName in momentumFiles:
        file = pd.read_csv(f"{mainDirectory}/Data/Storage/Raw/Momentum/{momentumFolder}/{fileName}", index_col="Date")
        file = pd.Series(file.iloc[:, 0])
        df.insert(0,fileName, file)

    dfi = pd.read_csv(f"{mainDirectory}/Data/Storage/Raw/Y/{yFilename}" , index_col="Date")
    dfi = pd.Series(dfi.iloc[:, 0])
    df.insert(0,yFilename[:-4],dfi )

    df.to_csv("{}/Data/In/{}".format(mainDirectory, nameForDataset))
    print(f"Dataset {nameForDataset} made!")

