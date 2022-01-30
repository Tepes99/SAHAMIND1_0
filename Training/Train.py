"""
Model training script with all the user inputs.
"""

import TrainingModel as train


train.startTrainingModels(
    trainingSetLabel            = "test",
    dataset                     = "test.csv",
    activations                 = ["relu", "sigmoid"],
    losses                      = ["mean_squared_error"],
    epochsList                  = [500],
    numberOfLSTMLayers          = [1,2],
    denseLayer                  = True,
    layerSizes                  = [12, 16],
    batchSizesList              = [46, 50],
    validationDataStartingDate  = "2021-01-04",
    earlyStop                   = True,
    patience                    = 200,
    )