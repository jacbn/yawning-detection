from yawnnlib.training import models
from yawnnlib.utils import config

import sys

def trainSpecificModels(models, sampleRate : int):
    assert config.get("TRAIN_SPLIT") == 1, "TRAIN_SPLIT should be 1 to train all models to ensure testing occurs on separate, consistent data."
    repeats = int(input("Specify how many times to repeat each model: "))
    for modelNum in models:
        model = modelMap[modelNum]
        for i in range(repeats):
            model(sampleRate, i, repeats)

modelMap = {
    1 : models.trainEimuLSTM,
    2 : models.trainEimuCNN,
    3 : models.trainEimuConvLSTM, 
    4 : models.trainFftLSTM,
    5 : models.trainFftCNN,
    6 : models.trainFftConvLSTM,
    7 : models.trainSpectrogramCNN, 
    8 : models.trainAlternativeClassifiers, 
    9 : lambda sr: trainSpecificModels(range(1, 9), sr),
    10 : lambda sr: trainSpecificModels(range(1, 4), sr),
    11 : lambda sr: trainSpecificModels(range(4, 8), sr),
}

if __name__ == "__main__":
    sampleRate = -1
    args = sys.argv[1:]
    try:
        # allow the user to enter the arguments they would use throughout the program immediately for faster use
        model = modelMap[int(args[0])]
        if len(args) > 1:
            sampleRate = int(args[1])
    except:
        # if the user did not enter the arguments, ask them to enter them
        print("Please specify a model to train:")
        print(" 1: Eimu LSTM")
        print(" 2: Eimu CNN")
        print(" 3: Eimu ConvLSTM")
        print(" 4: FFT LSTM")
        print(" 5: FFT CNN")
        print(" 6: FFT ConvLSTM")
        print(" 7: Spectrogram CNN")
        print(" 8: Alternative Models")
        print(" 9: All Models")
        print("10: All Eimu Models")
        print("11: All FFT Models")
        while True:
            try:
                model = modelMap[int(input())]
                break
            except:
                continue
            
        print("(Optional) Please specify a sample rate (Hz) to resample the data:")
        try:
            sampleRate = int(input())
        except:
            pass
    finally:
        model(sampleRate)