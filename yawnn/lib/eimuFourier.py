import commons
from eimuReader import SessionData

import numpy as np
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt

def eimuToFourier(eimuPath : str):
    session = SessionData.fromPath(eimuPath)
    data, timestamps = session.toRaw()
    # frequencies = np.zeros((len(data), 6))
    frequencies = [[None for _ in range(6)] for _ in range(len(data))]
    frequencies = np.empty((len(data), 6, 64))
    for i in range(len(data)):
        for axis in range(6):
            frequencies[i][axis] = fft(data[i][:,axis])
            # print(len(frequencies[i][axis]))
        # data[i] = fft(data[i])
    # print(frequencies[0])
    
    
    N = len(data[0])
    T = 1.0 / 32.0
    x = np.linspace(0.0, N*T, N, endpoint=False)
    xf = fftfreq(N, T)[:N//2]
    for i in range(6):
        y = data[0][:,i]
        yf = fft(y)
    
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2])) #type: ignore
    plt.grid()
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
    
    