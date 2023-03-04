from yawnnlib.training import models

import sys

modelMap = {
    1 : models.trainEimuLSTM,
    2 : models.trainEimuCNN,
    3 : models.trainEimuConvLSTM, 
    4 : models.trainFourierCNN, 
    5 : models.trainFourierLSTM,
    6 : models.trainTDFourierCNN, 
}

if __name__ == "__main__":
    sampleRate = -1
    args = sys.argv[1:]
    try:
        model = modelMap[int(args[0])]
        if len(args) > 1:
            sampleRate = int(args[1])
    except:
        print("Please specify a model to train:")
        print(" 1: Eimu LSTM")
        print(" 2: Eimu CNN")
        print(" 3: Eimu ConvLSTM")
        print(" 4: Fourier CNN")
        print(" 5: Fourier LSTM")
        print(" 6: Time-Distributed Fourier CNN")
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