from yawnnlib.utils import commons, filters
from yawnnlib.structure.sessionData import SessionData
from yawnnlib.structure.sensorReading import SensorReading
from yawnnlib.structure.timestamp import Timestamp

import numpy as np
from scipy.fft import rfft, rfftfreq, ifft
from scipy import signal
from matplotlib import pyplot as plt
from typing import Callable

class FourierData(SessionData):
    def __init__(self, dataset : list[SensorReading], timestamps : list[Timestamp], sampleRate : int, version : int, sessionID : int = -1, totalSessions : int = -1, nPerSeg : int = 128, nOverlap : int = 96):
        """ Initializes a FourierData object.

        Parameters
        ----------
        dataset : list[SensorReading]
            The data to be processed.
        timestamps : list[Timestamp]
            The timestamps for the data.
        sampleRate : int
            The sample rate of the data.
        version : int
            The version of the data.
        sessionID : int, optional
            The ID of a session when split, by default -1
        totalSessions : int, optional
            The total number of sessions in the split, by default -1
        nPerSeg : int, optional
            FFT parameter, used to determine width of FFT windows. ~256 is optimal, but has high runtime length and data size. By default 128
        nOverlap : int, optional
            FFT parameter, used to determine separation of windows. nPerSeg-1 is optimal, but at a significant runtime cost. By default 96
        """
        super().__init__(dataset, timestamps, sampleRate, version, sessionID, totalSessions)
        self.nPerSeg = nPerSeg
        self.nOverlap = nOverlap
    
    @classmethod
    def fromPath(cls, path : str, fileNum : int = -1, totalFiles : int = -1, nPerSeg : int = 128, nOverlap : int = 96):
        """ Creates a FourierData object from a .eimu file.

        Parameters
        ----------
        path : str
            The path to the .eimu file.
        fileNum : int, optional
            The current file number, by default -1
        totalFiles : int, optional
            The total number of files, by default -1
        nPerSeg : int, optional
            FFT parameter, used to determine width of FFT windows. ~256 is optimal, but has high runtime length and data size. By default 128
        nOverlap : int, optional
            FFT parameter, used to determine separation of windows. nPerSeg-1 is optimal, but at a significant runtime cost. By default 96

        Returns
        -------
        FourierData
            The FourierData object.
        """
        session = super().fromPath(path, fileNum, totalFiles)
        session.__class__ = cls
        session.nPerSeg = nPerSeg
        session.nOverlap = nOverlap
        return session
    
    def _applyToWindows(self, applyFunction : Callable[..., np.ndarray], dataFilter : filters.DataFilter = filters.NormalisationFilter(), windowSize : float = commons.YAWN_TIME*2, windowSep : float = commons.YAWN_TIME/2) -> tuple[np.ndarray, list[Timestamp]]:
        """ Splits the class data (self.accel, self.gyro) into windows, applies a given function to each, then returns a matrix of the results.

        Parameters
        ----------
        applyFunction : Callable[..., np.ndarray]
            The function to apply to each window.
        dataFilter : filters.DataFilter, optional
            The filter to apply to the data. All uses of this are for FFT so by default filters.NormalisationFilter()
        windowSize : float, optional
            The size of the windows to split the data into, by default commons.YAWN_TIME*2
        windowSep : float, optional
            The separation between windows, by default commons.YAWN_TIME/2

        Returns
        -------
        tuple[np.ndarray, list[Timestamp]]
            A pair of (data, timestamps). The data has the shape (axes, windows, {function result}).

        Raises
        ------
        ValueError
            If the data cannot be split into windows of the given size.
        """
        axisResults = []
        timestamps = []
        
        trueWindowSize = int(windowSize * self.sampleRate)
        trueWindowSep = int(windowSep * self.sampleRate)
        boundary = self.nPerSeg//2
          
        pString = f"  Calculating Fourier frequencies: "
        print(pString + "......", end='')
            
        for axis in range(6):
            # obtain and filter the data
            data = self._getDataByAxis(axis)
            dataFiltered = dataFilter.apply(data)
            
            # there won't be any spectrogram data outside of dataFiltered[boundary:-boundary] as this is the boundary required to calculate the fft
            if trueWindowSize > len(dataFiltered[boundary:-boundary]):
                raise ValueError(f"Not enough data to split into windows of {windowSize} seconds. Try lowering the window size, or using larger files.")
            
            # split the data into windows
            windowResults = []            
            windowStart = boundary

            while windowStart + trueWindowSize < len(dataFiltered) - boundary:
                window = dataFiltered[windowStart-boundary : windowStart+trueWindowSize+boundary]
                
                windowResult = applyFunction(window)

                windowResults.append(windowResult)
                if axis == 0:
                    # we add a positive timestamp for the spectrogram if the window contains the precise time of a yawn
                    # todo: control leeway as a hyperparameter
                    timestamps.append(len(self._getRelevantTimestamps(self.timestamps, windowStart, windowStart+trueWindowSize)) > 0)
                    
                windowStart += trueWindowSep
            
            axisResults.append(windowResults)
            
            print('\r' + pString + '#' * (axis+1) + '.' * (5-axis), end='' if axis < 5 else '\n')
        
        data = np.array(axisResults, dtype=np.float64)
        return data, timestamps
    
    def getFFTData(self, dataFilter : filters.DataFilter = filters.NormalisationFilter(), windowSize : float = commons.YAWN_TIME*2, windowSep : float = commons.YAWN_TIME/2) -> tuple[np.ndarray, list[Timestamp]]:
        """ Returns the magnitudes of the FFTs of the data split into windows. Note that the data **must** be normalised (i.e. use filters.NormalisationFilter()) before being processed.

        Parameters
        ----------
        dataFilter : filters.DataFilter, optional
            The filter to apply to the data, by default filters.NormalisationFilter()
        windowSize : float, optional
            The size of the windows to split the data into, by default commons.YAWN_TIME*2
        windowSep : float, optional
            The separation between windows, by default commons.YAWN_TIME/2

        Returns
        -------
        tuple[np.ndarray, list[Timestamp]]
            _description_
        """
        assert isinstance(dataFilter, filters.NormalisationFilter) or (isinstance(dataFilter, filters.FilterCollection) and any([map(lambda x: isinstance(x, filters.NormalisationFilter), dataFilter.filters)])), "Fourier data must be normalised before being processed. Use a NormalisationFilter or a FilterCollection containing a NormalisationFilter."

        # for each axis and window, calculate the FFT
        func = lambda x: self._getFFTMagnitudes(x)[1]                                        # gets the magnitudes of the FFT
        data, timestamps = self._applyToWindows(func, dataFilter, windowSize, windowSep) # apply this to each window
        
        ax, ch, fs = data.shape
        assert len(timestamps) == ch
        data = np.transpose(data, (1, 2, 0))
    
        # data format is (windows, frequencies, axes)
        return data, timestamps

    def getSpectrogramData(self, dataFilter : filters.DataFilter = filters.NormalisationFilter(), windowSize : float = commons.YAWN_TIME*2, windowSep : float = commons.YAWN_TIME/2) -> tuple[np.ndarray, list[Timestamp]]:
        """ Returns spectrogram data for the given input data.

        Parameters
        ----------
        dataFilter : filters.DataFilter, optional
            The filter to apply to the data (BEFORE applying FFT), by default filters.NormalisationFilter()
        windowSize : float, optional
            The window size in seconds, by default commons.YAWN_TIME*2
        windowSep : float, optional
            The separation between windows in seconds, by default commons.YAWN_TIME/2

        Returns
        -------
        tuple[np.ndarray, list[Timestamp]]
            The data and timestamps for the windows.

        Raises
        ------
        ValueError
            If the data is too small to be split into windows of windowSize seconds.
        """
        
        assert isinstance(dataFilter, filters.NormalisationFilter) or (isinstance(dataFilter, filters.FilterCollection) and any([map(lambda x: isinstance(x, filters.NormalisationFilter), dataFilter.filters)])), "Fourier data must be normalised before being processed. Use a NormalisationFilter or a FilterCollection containing a NormalisationFilter."
        
        # for each axis and window, calculate the spectrogram
        func = lambda x: self._getSpectrogram(x)[2]                                          # gets the Sxx data from _getSpectrogram
        data, timestamps = self._applyToWindows(func, dataFilter, windowSize, windowSep) # apply this to each window
        
        ax, ch, fs, ts = data.shape
        assert len(timestamps) == ch
        data = np.transpose(data, (1, 3, 2, 0))
        
        # data format is (windows, times (samples) per window, frequencies, axes)
        return data, timestamps
    
    def _getDataByAxis(self, axis : int):
        return np.array(list(map(lambda x: x[axis%2][axis//2], zip(self.accel, self.gyro))))
        
    def plotSessionData(self, show : bool = False, figure : int = 1, dataFilter : filters.DataFilter = filters.NoneFilter()) -> None:
        for axis in range(6):
            data = self._getDataByAxis(axis)
            dataFiltered = dataFilter.apply(data)
            
            self.plot(show=False, figure=figure, unitConversion=False)
            self._plotFFTMagnitudes(dataFiltered, axis, figure+1, False)
            self._plotIFFTReconstruction(dataFiltered, axis, figure+2, False)
            self._plotSpectrograms(dataFiltered, axis, figure+3, False, fmin=0, fmax=6, maxAmp=-1)           
        
        if show:
            plt.show()
                    
    def _getFFTMagnitudes(self, data : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # we use abs here as we only care about the magnitude
        fourierData = np.abs(rfft(data, norm='ortho')) # type: ignore 
        N = len(fourierData)
        xf = rfftfreq(N*2-1, 1/self.sampleRate)
        return xf, fourierData
    
    def _getSpectrogram(self, data : np.ndarray):
        f, t, Sxx = signal.spectrogram(data, self.sampleRate, nperseg=self.nPerSeg, noverlap=self.nOverlap)
        return f, t, Sxx
    
    def _plotFFTMagnitudes(self, data : np.ndarray, axis : int, figure : int = 2, show : bool = False) -> None:
        plt.figure(figure)
        plt.suptitle("FFT Magnitudes")
        ax = plt.subplot(3,2,axis+1)
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2])
        
        xf, fourierData = self._getFFTMagnitudes(data)
    
        # plots the magnitude of all frequencies
        ax.set_yscale("log")
        ax.stem(xf, fourierData, 'r', markerfmt=' ')
        if (show):
            plt.show()
            
    def _plotIFFTReconstruction(self, data : np.ndarray, axis : int, figure : int = 3, show : bool = False) -> None:
        plt.figure(figure)
        plt.suptitle("Inverse FFT Reconstructions")
        ax = plt.subplot(3,2,axis+1)
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left')
        
        # here we do care about the sign, so we don't use abs
        fourierData = rfft(data, norm='ortho')
        reconstructedData = np.array(ifft(fourierData, norm='ortho'))
        
        # renormalise the data
        reconstructedData = filters.NormalisationFilter().apply(reconstructedData)
                
        # plots the reconstruction of the frequencies
        ax.plot(np.arange(len(reconstructedData))*2, reconstructedData, color=commons.AXIS_COLOURS[axis//2]) 
        
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left')
        ax.set_ylabel("Acceleration (m/s^2)" if axis//2 == 0 else "Angular Velocity (deg/s)")
        ax.set_xlabel(f"Samples ({self.sampleRate} = 1 sec)")
        
        for timestamp in self.timestamps:
            ax.axvline(timestamp.time, color='black', alpha=0.5)
            
        if (show):
            plt.show()
            
    def _plotSpectrograms(self, data : np.ndarray, axis : int, figure : int = 5, show : bool = False, fmin : int = 0, fmax : int = 6, maxAmp : int = -1) -> None:
        plt.figure(figure)
        plt.suptitle("Axis Spectrograms")
        ax = plt.subplot(3,2,axis+1)
        plt.tight_layout(pad=-0.5)
        f, t, Sxx = signal.spectrogram(data, self.sampleRate, nperseg=self.nPerSeg, noverlap=self.nOverlap)
        
        # plot only the frequencies in the given range
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        f = f[freq_slice]
        Sxx = Sxx[freq_slice,:][0] # type: ignore
        
        if maxAmp > 0:
            Sxx[Sxx > maxAmp] = np.nan
        
        pc = ax.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.colorbar(pc)
        
        for timestamp in self.timestamps:
            ax.axvline(timestamp.time/self.sampleRate, color='black', alpha=0.5)
            
        ax.set_title(commons.AXIS_NAMES[axis%2][axis//2], loc='left', fontsize=20)
        ax.set_ylabel('Frequency [Hz]', fontsize=16)
        ax.set_xlabel('Time [sec]', fontsize=16)
        
        if (show):
            plt.show()
    
    
if __name__ == "__main__":
    # s = FourierData.fromPath(f"{commons.PROJECT_ROOT}/data/tests/96hz/96hz-yawns1.eimu")
    s = FourierData.fromPath(f"{commons.PROJECT_ROOT}/data/user_trials/PRESENTATION/john.eimu")
    
    s.plot(show=False, figure=1, unitConversion=True)
    # s = FourierData.applyFilter(s, filters.HighPassFilter(96, 0.1), filters.ApplyType.SESSION)
    # s = FourierData.applyFilter(s, filters.LowPassFilter(96, 8, order=3), filters.ApplyType.SESSION)
    # s = FourierData.applyFilter(s, filters.NormalisationFilter(), filters.ApplyType.SESSION)
    assert isinstance(s, FourierData)
    s.plotSessionData(show=False, figure=8)
    plt.show()
    