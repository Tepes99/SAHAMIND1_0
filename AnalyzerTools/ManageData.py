"""
You can use these scripts to manage data. It is possible to run all the scripts in succession or just one in particular, like the 'makeDataset' function
"""
import DataManagement as dmp
import pandas as pd 
import os

mainDirectory = os.path.dirname(os.path.abspath(__file__))
mainDirectory = mainDirectory[0:-14]

dmp.dowloadPriceData(
    (2003,12,30), 
    (2022,1,29), 
    [
        "CL=F",
        "CNY=X",
        "EURUSD=X",
        "GC=F",
        "JPY=X",
        "^GSPC",
        "^IRX",
        "^TNX",
        "^TYX",
        "^VIX"
    ]
)

dmp.runQM("test")

dmp.returns(21, "^GSPC.csv", "^GSPC.csv")

dmp.makeDataset(
    nameForDataset="test.csv",
    yFilename="^GSPC.csv",
    momentumFolder= "test",
    momentumFiles= [
        "CL=F.csv",
        "CNY=X.csv",
        "EURUSD=X.csv",
        "GC=F.csv",
        "JPY=X.csv",
        "^GSPC.csv",
        "^IRX.csv",
        "^TNX.csv",
        "^TYX.csv",
        "^VIX.csv"
        ]
)

