import commons
from eimuReader import SessionData

import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

SIGNIFICANT_FREQ = 0.1
SAMPLE_RATE = 32

def eimuToFourier(eimuPath : str):
    session = SessionData.fromPath(eimuPath)
    data, timestamps = session.toRaw()
    # frequencies = np.zeros((len(data), 6))
    frequencies = [[None for _ in range(6)] for _ in range(len(data))]
    # frequencies = np.empty((len(data), 6, 64))
    for i in range(len(data)):
        for axis in range(6):
            frequencies[i][axis] = fft(data[i][:,axis]) # type: ignore
            # print(len(frequencies[i][axis]))
        # data[i] = fft(data[i])
    # print(frequencies[0])
    
    session.plot(show=False)
    
    sample = 375
    
    N = len(data[sample])
    T = 1 / SAMPLE_RATE
    x = np.linspace(0, N*T, N, endpoint=False)
    xf = fftfreq(N, T)[:N//2] 
    for i in range(6):
        y = data[sample][:,i]
        yf = fft(y) # the first N//2 elements are the positive frequencies; the rest are negative, but are symmetric if we need to calculate
        
        posF = 2/N * np.abs(yf[:N//2])
        
        significant = list(map(lambda y: xf[y[0]], filter(lambda x: x[1] > SIGNIFICANT_FREQ, enumerate(posF))))
        print(significant)
    
        plt.plot(xf, posF) 
        
    plt.grid()
    plt.legend(["ax", "ay", "az", "gx", "gy", "gz"], loc="upper right")
    plt.show()
    
    # plt.plot(xt, frequencies[0][0])

def directoryToFourierInput(path : str) -> tuple[tuple, tuple]:
    inputs = commons.mapToDirectory(eimuToFourier, path)
    # combine all the inputs. each is a tuple of (trainX, trainY), (testX, testY),
    # and the result is a combination of all the trainX, trainY, testX, testY individually
    return ((),())
    
if __name__ == "__main__":
    # directoryToFourierInput("./yawnn/data")
    eimuToFourier("./yawnn/data/long3.eimu")
    
    