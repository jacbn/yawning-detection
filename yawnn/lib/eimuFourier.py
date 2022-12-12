import commons
from eimuReader import SessionData

from scipy.fft import fft

def eimuToFourier(eimuPath : str):
    session = SessionData.fromPath(eimuPath)
    data, timestamps = session.toRaw()
    print(data)
    
if __name__ == "__main__":
    eimuToFourier("./yawnn/data")