'''
Created on Jun 19, 2012

@author: bogdan
'''
import ufft.fft_utils as fft_utils
import mlpy.wavelet as wave
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class Cwt(object):
    '''
    Python class for wavelet analysis and the statistical approach
    using Mashine Language Python wavelet implementation
    
    API Not completely tested. Main functionality tested below in __main__ 
    '''


    def __init__(self, path, file1, file2, tunits = "day"):
        '''
        Constructor
        '''

        if path != None and file1 != None:
            self.filename1 = file1
        if file2 != None:
            self.filename2 = file2
        else:
            self.filename2 = None

        if path != None and file1 != None:
            #read Lake data
            [self.Time, self.SensorDepth1] = fft_utils.readFile(path, file1)
            eps = (self.Time[1] - self.Time[0]) / 100
            self.SensorDepth1 = np.array(self.SensorDepth1)
            self.Time = np.array(self.Time)
            if self.Time[0] < 695056:
                self.Time += 695056


        #read bay data
        if file2 != None:
            [self.Time1, y1] = fft_utils.readFile(path, file2)
            #resample to be the same as first only if needed
            if (self.Time[1] - self.Time[0]) - (self.Time1[1] - self.Time1[0]) > eps:
                y1 = signal.resample(y1, len(self.Time))
                self.SensorDepth2 = np.array(y1)
            else:
                self.SensorDepth2 = y1
            self.SensorDepth2 = np.array(self.SensorDepth2)
            self.Time1 = np.array(self.Time1)

    def doSpectralAnalysis(self, motherW = 'morlet'):
        #[self.Time, SensorDepth1, X1, scales1, freq1, corr1 ] 
        a = self.doSpectralAnalysisOnSeries(motherW = 'morlet', no = "1")
        #[self.Time, SensorDepth2, X2, scales2, freq2, corr2 ]
        if self.filename2 != None:
            b = self.doSpectralAnalysisOnSeries(motherW = 'morlet', no = "2")
        else:
            b = None
        return [a, b]

    def doSpectralAnalysisOnSeries(self, motherW = 'morlet', no = "1"):

        SensorDepth = eval('self.SensorDepth' + no)

        if motherW == 'dog':
            omega0 = 2.0
        elif motherW == 'morlet':
            omega0 = 6.0

        dt = (self.Time[1] - self.Time[0]) * 24 # time is in days , convert to hours

        L = SensorDepth.shape[0]
        scales = wave.autoscales(N = L, dt = dt, dj = 0.05, wf = motherW, p = omega0)

        # 'dt' is a time step in the time series
        signal = mlab.detrend_linear(SensorDepth)

        X = wave.cwt(x = signal, dt = dt, scales = scales, wf = motherW, p = omega0)
        freq = wave.fourier_from_scales(scales, motherW, omega0)
        self.freq = freq / 3600 # 1. / freq

        return [self.Time, SensorDepth, X, scales, freq]

    def getLakeTransform(self):
        return self.Y

    def getBayTransform(self):
        return self.Y1

    def getTime(self):
        return self.Time


if __name__ == '__main__':
    '''
    Testing ground for local functions

    '''
    #1Test the simple scalogram for Dog and Morlet wavelets

    x = np.random.sample(512)
    scales = wave.autoscales(N = x.shape[0], dt = 1, dj = 0.05, wf = 'morlet', p = 6)
    X = wave.cwt(x = x, dt = 1, scales = scales, wf = 'morlet', p = 6)
    fig = plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    p1 = ax1.plot(x)
    ax1.set_title("Random 512 points, DOG wavelet")
    ax1.autoscale_view(tight = True)
    ax2 = plt.subplot(2, 1, 2)
    p2 = ax2.imshow(np.abs(X), interpolation = 'nearest')

    #2) Test true amplitude
    Fs = 1000.0                     # Sampling frequency
    T = 1.0 / Fs                    # Sample time
    L = 1024                         # Length of signal
    t = np.array(list(range(0, L))) * T                # Time vector
    # Sum of a 50 Hz sinusoid and a 120 Hz sinusoid
    x = np.array([])
    x1 = 0.7 * np.sin(2 * np.pi * 40 * t)
    def fun(t):
        x = np.zeros(len(t))
        for i in range(0, len(t)):
            if i < len(t) / 2:
                x[i] = 0.0 * np.sin(2 * np.pi * 120 * t[i])
            else:
                x[i] = 4.0 * np.sin(2 * np.pi * 120 * t[i])
        return x
    #x2 = 4.0 * np.sin(2 * np.pi * 120 * t)
    x2 = fun(t)
    x3 = 8.0 * np.sin(2 * np.pi * 200 * t)
    x4 = 6.0 * np.sin(2 * np.pi * 400 * t)
    title = 'Signal Corrupted with Zero-Mean Random Noise'
    xlabel = 'time (milliseconds)'
    x = (x1 + x2 + x3 + x4)


    # normalize by standard deviation (not necessary, but makes it easier
    # to compare with plot on Interactive Wavelet page, at
    # "http://paos.colorado.edu/research/wavelets/plot/"
    std = np.std(x)
    variance = std ** 2
    #x = (x - np.mean(x)) / np.sqrt(variance)
    #x = x * np.sqrt(2 / np.pi) / std
    signal = x
    omega0 = 6 #default value
    dt = t[1] - t[0]

    wavelet = 'morlet'
    prec = 0.01
    scales = wave.autoscales(N = signal.shape[0], dt = dt, dj = prec, wf = wavelet, p = omega0)

    X = wave.cwt(x = signal, dt = dt, scales = scales, wf = wavelet, p = omega0)

    #plot spectrogram 
    # frequency works for "morlet" only here f = 1 / lambda  ( wavelength)
    # freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales[1:])
    freq = wave.fourier_from_scales(scales, wavelet, omega0)
    freq = 1. / freq

    #window = 'hanning'
    #w = eval('np.' + window + '(freq.shape[0])')
    #amp = np.abs(X.conj().transpose() * w)
    power = (np.abs(X)) ** 2
    amp = np.abs(X) / 2.
    #amp = np.sqrt(2 * power / L / 0.375)
    phase = np.angle(X)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.7, 0.60])
    ax3 = fig1.add_axes([0.83, 0.1, 0.03, 0.6])
    #ax1.set_yscale('log')
    im1 = ax1.pcolormesh(t, freq, amp)

    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 0.7, 0.60])
    ax4 = fig2.add_axes([0.83, 0.1, 0.03, 0.6])
    #ax2.set_yscale('log')
    im2 = ax2.pcolormesh(t, freq, phase)

    # set correct way of axis, whitespace before and after with window
    # length
    ax1.axis('tight')
    # ax.set_xlim(0, end)
    ax1.grid(False)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Amplitude')
    fig1.colorbar(im1, cax = ax3)

    # set correct way of axis, whitespace before and after with window
    # length
    ax2.axis('tight')
    # ax.set_xlim(0, end)
    ax2.grid(False)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title('Phase')
    fig2.colorbar(im2, cax = ax4)





    fig3 = plt.figure()
    plt.title('Global Wavelet Spectrum Amplitude: np.sum(power, axis = 1) / L')
    A = np.sum(amp, axis = 1) / L
    #A2 = variance * np.sum(power, axis = 1) / (L)
    plt.plot(freq, A)
    #plt.show()

    #fig4 = plt.figure()
    #correction factor -empirical and I don't trust it
    #
    # plt.title('Global Wavelet Spectrum Corrected Amplitude: np.sum(power, axis = 1) / L')
    # fc = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scales) * np.sqrt(scales) / (omega0 * np.pi * np.pi / 2)
    # plt.plot(freq, A * fc)

    plt.show()





