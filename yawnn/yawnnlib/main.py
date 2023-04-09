from yawnnlib.training import models
from yawnnlib.utils import config

import sys

def trainAllModels(sampleRate : int = -1):
    assert config.get("TRAIN_SPLIT") == 1, "TRAIN_SPLIT should be 1 to train all models to ensure testing occurs on separate, consistent data."
    repeats = int(input("Specify how many times to repeat each model: "))
    for model in modelMap.values():
        for _ in range(repeats):
            if sampleRate < 1:
                model()
            else:
                model(sampleRate)

modelMap = {
    1 : models.trainEimuLSTM,
    2 : models.trainEimuCNN,
    3 : models.trainEimuConvLSTM, 
    4 : models.trainFftLSTM,
    5 : models.trainFftCNN,
    6 : models.trainFftConvLSTM,
    7 : models.trainSpectrogramCNN, 
    8 : models.trainAlternativeClassifiers, 
    9 : trainAllModels,
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
        if sampleRate < 1:
            model()
        else:
            model(sampleRate)