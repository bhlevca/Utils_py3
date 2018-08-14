'''
Created on Jun 11, 2012

@author: bogdan
'''
# local imports

import numpy as np
import scipy as sp
from . import fft_utils
from . import filters
from . import peakdetect
from . import peakdek
from . import Filter

# system imports
import time, os
import csv
import math
from scipy import fftpack
import scipy.signal
import matplotlib.mlab
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import *
import utools.phase_shift as phase_shift

path = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/LakeOntario-data'
path1 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/FMB'
path2 = 'software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/LOntario'


class FFTSpectralAnalysis(object):
    '''
    classdocs
    '''

    def __init__(self, path, file1, file2, data = None, data1 = None):
        '''
        Constructor
        '''
        if file1 != None:
            self.path_in = path
            self.filename = file1
            self.filename1 = file2
            self.data = None
            self.data1 = None
        else:
            self.data = data
            self.data1 = data1
            self.path_in = None
            self.filename = None
            self.filename1 = None
        # END IF

    # end


    def FourierAnalysis(self, filename, draw, tunits = 'sec', window = 'hanning', num_segments = 1, filter = None, \
                        log = 'linear', bResample = False, Time1 = None, date1st = False, dateinterval=None, d3d=False):

        '''
        FFT for Spectral Analysis
        =========================

        This example shows the use of the FFT function for spectral analysis.
        A common use of FFT's is to find the frequency components of a signal buried in a noisy
        time domain signal.

        First create some data. Consider data sampled at every 5 min for Lake Ontario and every  3 min in Frechman's bay.
        '''

        # read Lake data
        [Time, SensorDepth] = fft_utils.readFile(self.path_in, filename, date1st)
        if dateinterval != None:
            print ("Original time series 1: %d", len(SensorDepth) )
            Time, SensorDepth = fft_utils.select_interval_data(Time, SensorDepth, dateinterval)
            print ("Reduced time series 2: %d", len(SensorDepth) )
      
        #if filename == ' 01-13-Aug-WL-Spadina_out.csv':
        #        print("EXCEPTION for model data: delay artificially by 36 minutes to match real data phase!")
        #        SensorDepth=shift.phase_shift(12,SensorDepth)
                
        Time = np.array(Time)
        SensorDepth = np.array(SensorDepth)
        if bResample:
            print("*** Resampling SensorDepth and Time from length: %d to length: %d" % (len(Time), len(Time1)))
            # SensorDepth, spos = sp.signal.resample(SensorDepth, len(Time1), SensorDepth)
            # Time, tpos = sp.signal.resample(Time, len(Time1), Time)
            from matplotlib.dates import date2num, num2date
            dat = []
            dat1 = []
            i = 0
            for t in Time:
                if t < 695056:
                    t += 695056
                datet = num2date(t)
                dat.append(datet)
                i += 1

            i = 0
            for t1 in Time1:
                if t1 < 695056:
                    t1 += 695056
                dat1.append(num2date(t1))
                i += 1

            ts = pandas.Series(SensorDepth, index = dat)
            ts2 = ts.reindex(dat1, method = 'ffill')
            SensorDepth = ts2.values
            SensorDepth[0] = SensorDepth[1]
            Time = Time1


        # prepare for the amplitude spectrum analysis
        if tunits == 'day':
            factor = 86400
        elif tunits == 'hour':
            factor = 3600
        else:
            factor = 1
        dt_s = (Time[2] - Time[1]) * factor  # Sampling period [s]
        Fs = 1 / dt_s  # Samplig freq    [Hz]

        # Filter data here on the whole lenght
        if filter != None:
            lowcut = filter[0]
            highcut = filter[1]
            btype = 'band'
            y, w, h, b, a = filters.butterworth(SensorDepth, btype, lowcut, highcut, Fs, recurse = True)
            # filter.butterworth(SensorDepth, Fs)
            SensorDepth = y
            # ToDo:
            # here we can enable some filter display and test
        # end filter

        x05 = None
        x95 = None

        if Time[0] < 695056:
            Time += 695056
        if num_segments == 1:
            [y, Time, fftx, NumUniquePts, mx, f, power] = self.fourierTSAnalysis(Time, SensorDepth, draw, tunits)  # , window, filter)
        else:
            [f, avg_fftx, avg_amplit, avg_power, x05, x95] = self.WelchFourierAnalysis_overlap50pct(Time, SensorDepth, draw, tunits, window, num_segments, log)
            fftx = avg_fftx
            mx = avg_amplit
            power = avg_power
            y = sp.signal.detrend(SensorDepth)
            NFFT = len(Time)
            NumUniquePts = int(math.ceil((NFFT / 2) + 1))

        return [SensorDepth, y, Time, fftx, NumUniquePts, mx, f, power, x05, x95]

    # end

    def FourierDataAnalysis(self, data, showOrig, draw, tunits = 'sec', window = 'hanning', num_segments = 1, log = 'linear', date1st = False):

        Time, SensorDepth = data
        x05 = None
        x95 = None

        if Time[1] < 695056:
            Time += 695056

        if num_segments == 1:
            [y, Time, fftx, NumUniquePts, mx, f, power] = self.fourierTSAnalysis(Time, SensorDepth, draw, tunits)
        else:
            [f, avg_fftx, avg_amplit, avg_power, x05, x95] = self.WelchFourierAnalysis_overlap50pct(Time, SensorDepth, draw, tunits, window, num_segments, log)
            fftx = avg_fftx
            mx = avg_amplit
            power = avg_power
            y = sp.signal.detrend(SensorDepth)
            NFFT = len(Time)
            NumUniquePts = int(math.ceil((NFFT / 2) + 1))
        # end if

        return [SensorDepth, y, Time, fftx, NumUniquePts, mx, f, power, x05, x95]


    def fourierTSAnalysis(self, Time, SensorDepth, draw = 'True', tunits = 'sec', log = 'linear'):
        '''
        Clearly, it is difficult to identify the frequency components from looking at this signal;
        that's why spectral analysis is so popular.

        Finding the discrete Fourier transform of the noisy signal y is easy; just take the
        fast-Fourier transform (FFT).

        Compute the amplitude spectral density, a measurement of the amplitude at various frequencies,
        using module (abs)

        Compute the power spectral density, a measurement of the energy at various frequencies,
        using the complex conjugate (CONJ).

        nextpow2 finds the exponent of the next power of two greater than or equal to the window length (ceil(log2(m))), and pow2 computes the power. Using a power of two for the transform length optimizes the FFT algorithm, though in practice there is usually little difference in execution time from using n = m.
        To visualize the DFT, plots of abs(y), abs(y).^2, and log(abs(y)) are all common. A plot of power versus frequency is called a periodogram:
        @param Time : the time points
        @param SensorDepth: the depth data timeseries
        @param draw: boolean - if True Plots additional Graphs
        @param tunits:
        @param window: = 'blackman' #NO; 'bartlett' #OK; 'hamming' #OK; 'hann' #BEST default; flattop; gaussian; blackmanharris; barthann; bartlett;
        @param num_segments = 1 default represents tne number of non overlapping segments used for the overlapping Welch method.
                              The number for the total number of ssegments is M = 2* num_segments-1: For example if num_segments=2 => M=3
        @param filter: defaul = None to avoid filtering twice in a recursive method. filtes is of type ufft.Filter

        @return: y             - detrended water levels
                 Time          - Time data points
                 fftx          - unique FFT values for the series
                 NumUniquePts  - size of the unique FFT values
                 mx            - the value of the single-sided FFT amplitude
                 f             - linear frequency vector for the mx points
        '''
        eps = 1e-3

        L = len(Time)

        # plot the original Lake oscillation input
        if draw:
            xlabel = 'Time [days]'
            ylabel = 'Z(t) [m]'
            legend = ['L. Ontario water levels']
            fft_utils.plotTimeSeries("Lake levels", xlabel, ylabel, Time, SensorDepth, legend)
        # end

        # prepare for the amplitude spectrum analysis
        if tunits == 'day':
            factor = 86400
        elif tunits == 'hour':
            factor = 3600
        else:
            factor = 1

        dt_s = (Time[2] - Time[1]) * factor  # Sampling period [s]
        Fs = 1 / dt_s  # Samplig freq    [Hz]

        # nextpow2 = This function is useful for optimizing FFT operations, which are most efficient when sequence length is an exact power of two.
        #  does seem to affect the value of the amplitude
        # # NFFT = ufft.nextpow2(L)   # Next power of 2 from length of the original vector, transform length
        #
        NFFT = L
        # DETREND THE SIGNAL is necessary to put all signals oscialltions around 0 ant the real level in spectral analysis
        yd = sp.signal.detrend(SensorDepth)

        # Take ufft, padding with zeros so that length(fftx) is equal to nfft
        fftx = fftpack.fft(yd, NFFT)  # DFT
        sFreq = np.sum(abs(fftx) ** 2) / NFFT
        sTime = np.sum(yd ** 2)

        # This is a sanity check
        assert abs(sFreq - sTime) < eps

        # What is power of the DFT and why does not show anything for us?
        # The FFT of the depth non CONJ shows good data except the beginning due to the fact
        # that the time series are finite.
        power = (np.abs(fftx)) ** 2  # Power of the DFT

        # TEST the Flat Top
        # amp = np.sqrt(2 * power / NFFT / Fw)
        # amp = 2 * np.abs(self.Wind_Flattop(fftx)) / NFFT

        # Calculate the number of unique points
        NumUniquePts = int(math.ceil((NFFT / 2) + 1))

        # FFT is symmetric, throw away second half
        fft2 = fftx[0:NumUniquePts]
        power = power[0:NumUniquePts]
        # amp = amp[0:NumUniquePts]


        # Take the magnitude of ufft of x and scale the ufft so that it is not a function of % the length of x
        # NOTE: If the frequency of interest is not represented exactly at one of the discrete points
        #       where the FFT is calculated, the FFT magnitude is lower.
        #
        # mx = np.abs(fftx.real) # was NumUniquePts or L but Numpy does normalization #
        # Since we dropped half the FFT, we multiply mx by 2 to keep the same energy.
        # The DC component and Nyquist component, if it exists, are unique and should not
        # be multiplied by 2.
        mx = 2 * np.abs(fft2) / NumUniquePts

        # This is an evenly spaced frequency vector with NumUniquePts points.
        # generate a freq spectrum from 0 to Fs / 2 (Nyquist freq) , NFFT / 2 + 1 points
        # The FFT is calculated for every discrete point of the frequency vector described by
        freq = np.array(list(range(0, NumUniquePts)))
        freq = freq * Fs / NFFT  # 2
        # same as
        # freq = np.ufft.fftfreq(NFFT, d = dt_s)[:NumUniquePts]

        return [yd, Time, fft2, NumUniquePts, mx, freq, power]
    # end

    def Wind_Flattop(self, Spec):
        '''
        Given an input spectral sequence 'Spec', that is the
        FFT of some time sequence 'x', Wind_Flattop(Spec)
        returns a spectral sequence that is equivalent
        to the FFT of a flattop windowed version of time
        sequence 'x'. The peak magnitude values of output
        sequence 'Windowed_Spec' can be used to accurately
        estimate the peak amplitudes of sinusoidal components
        in time sequence 'x'.



        Input: 'Spec' (a sequence of complex FFT sample values)
        Output: Windowed_Spec - a flattop windowed FFT transform
        Based on Lyons': "Reducing FFT Scalloping Loss Errors
        Without Multiplication", IEEE Signal Processing Magazine,
        '''
        N = len(Spec)
        Windowed_Spec = np.zeros(N, dtype = complex)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Perform convolution
        g_Coeffs = np.array([1, -0.94247, 0.44247], dtype = complex)
        # compute convuluted spce samples
        Windowed_Spec[0] = g_Coeffs[2] * Spec[N - 2] + g_Coeffs[1] * Spec[N - 1] + Spec[0] + g_Coeffs[1] * Spec[1] + g_Coeffs[2] * Spec[2]
        Windowed_Spec[1] = g_Coeffs[2] * Spec[N - 1] + g_Coeffs[1] * Spec[0] + Spec[0] + g_Coeffs[1] * Spec[2] + g_Coeffs[2] * Spec[3]

        # %compute last two
        Windowed_Spec[N - 2] = g_Coeffs[2] * Spec[N - 2] + g_Coeffs[1] * Spec[N - 1] + Spec[N - 2 ] + g_Coeffs[1] * Spec[N - 1] + g_Coeffs[2] * Spec[0]
        Windowed_Spec[N - 1] = g_Coeffs[2] * Spec[N - 3] + g_Coeffs[1] * Spec[N - 2] + Spec[N - 1] + g_Coeffs[1] * Spec[1] + g_Coeffs[2] * Spec[1]

        # Compute convolved spec samples for the middle of the spectrum
        for K in  range(2, N - 3):
            Windowed_Spec[K] = Spec[K] + g_Coeffs[1] * (Spec[K - 1] + Spec[K + 1]) + g_Coeffs[2] * (Spec[K - 2] + Spec[K + 2])

        # end %
        return Windowed_Spec
    # End of 'Wind_Flattop(Spec)' function


    #===============================================================================
    #
    #===============================================================================

    def WelchFourierAnalysis_overlap50pct(self, Time, SensorDepth, draw = False, tunits = "sec", window = 'hanning', nseg = 1, log = 'linear'):
        '''
            @param nseg: number of non overlapping segments. In calculation the total number of 50% overlapping segments K = N/M = 2*nseg-1
            @param Time : the time points
            @param SensorDepth: the depth data timeseries
            @param draw: boolean - if True Plots additional Graphs
            @param tunits: time units
            @param window: = 'blackman' #NO; 'bartlett' #OK; 'hamming' #OK; 'hann' #BEST default; flattop; gaussian; blackmanharris; barthann; bartlett;

            @return: y             - detrended water levels
                     Time          - Time data points
                     fftx          - unique FFT values for the series
                     NumUniquePts  - size of the unique FFT values
                     mx            - the value of the single-sided FFT amplitude
                     f             - linear frequency vector for the mx points
                     ph            - phase vector for the mx points
        '''
        den = 2 * nseg
        N = len(Time)
        M = int(N / nseg)
        t = np.zeros((den - 1, M))  # time segments
        x = np.zeros((den - 1, M))  # data segments

        for i in range(0, den - 1):
            fct = int(N / den)
            LInt = i * fct
            RInt = LInt + M
            tt = Time[LInt:RInt]
            xx = SensorDepth[LInt:RInt]
            t[i] = tt
            x[i] = xx
        # end for

        # perform FFT
        y = np.zeros((den - 1, M), dtype = np.float)  # data segments
        Tm = np.zeros((den - 1, M), dtype = np.float)  # time segments
        fftx = np.zeros((den - 1, int(M / 2) + 1), dtype = np.complex)  # transform segments
        NumUniquePts = np.zeros(den - 1)  # point segments
        amplit = np.zeros((den - 1, int(M / 2) + 1), dtype = np.float)  # amplit segments
        f = np.zeros((den - 1, int(M / 2) + 1), dtype = np.float)  # freq segments
        power = np.zeros((den - 1, int(M / 2) + 1), dtype = np.complex)  # power segments

        for i in range(0, den - 1):
            a = self.fourierTSAnalysis(t[i], x[i], draw, tunits, window)
            [y[i], Tm[i], fftx[i], NumUniquePts[i], amplit[i], f[i], power[i]] = a

        avg_amplit = 0
        avg_power = 0
        avg_y = 0
        avg_fftx = 0
        # calculate the average values
        for i in range(0, den - 1):
            avg_amplit += amplit[i]
            avg_power += power[i]
            # avg_y += y[i]
            avg_fftx += fftx[i]
        avg_amplit /= den - 1
        avg_power /= den - 1
        # avg_y /= den - 1 <= pointless
        avg_fftx /= den - 1

        interval_len = len(Time) / nseg
        data_len = len(Time)
        edof = fft_utils.edof(avg_amplit, data_len, interval_len, window)  # one dt chunk see Hartman Notes ATM 552 page 159 example

        (x05, x95) = fft_utils.confidence_interval(avg_amplit, edof, 0.95, log)

        return [f[0], avg_fftx, avg_amplit, avg_power, x05, x95]



if __name__ == '__main__':

    test = False
    if test == True:
        # 1) test plotting
        y1 = np.array([1, 2, 3])
        y2 = np.array([2, 3, 6])
        dt1 = datetime.strptime("2006-10-06 16:30", "%Y-%m-%d %H:%M")
        dt2 = datetime.strptime("2006-11-06 16:30", "%Y-%m-%d %H:%M")
        dt3 = datetime.strptime("2006-12-06 16:30", "%Y-%m-%d %H:%M")
        x1 = np.array([dt1, dt2, dt3])
        x2 = np.array([dt1, dt2, dt3])
        x_arr = np.array([x1, x2])
        y_arr = np.array([y1, y2])
        fftsa = FFTSpectralAnalysis(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv')
        fft_utils.plot_n_TimeSeries("Test", "labelx", "labely", x_arr, y_arr)
        fft_utils.plotTimeSeries("Test one", "labelx", "labely", x2, y2)

   # 2) Test FFT
    fftsa = FFTSpectralAnalysis(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv')
    draw = True

    # [y, Time, fftx, NumUniquePts, mx, f] = fftsa.FourierAnalysis(fftsa.filename, draw)
    # ufft.plotArray("FFT", "freq", "Amplit", f, mx)

    # 3) Test true amplitude
    Fs = 1000.0  # Sampling frequency
    T = 1.0 / Fs  # Sample time
    L = 100  # Length of signal
    t = np.array(list(range(0, L))) * T  # Time vector
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
    # x2 = 4.0 * np.sin(2 * np.pi * 120 * t)
    x2 = fun(t)
    x3 = 8.0 * np.sin(2 * np.pi * 200 * t)
    x4 = 6.0 * np.sin(2 * np.pi * 400 * t)
    title = 'Signal Corrupted with Zero-Mean Random Noise'
    xlabel = 'time (milliseconds)'
    x = (x1 + x2 + x3 + x4)
    fft_utils.plotArray(title, xlabel, "Amplit", Fs * t[1:500], x[1:500])
    fft_utils.plotArray(title, xlabel, "Amplit", Fs * t, x)

    window = 'hanning'
    # window = 'flattop_convol'
    # window = 'flattop'
    # window = 'flat'
    # window = 'kaiser'
    # window = None
    filter = None
    [y, Time, fftx, NumUniquePts, mx, f, power] = fftsa.fourierTSAnalysis(t, x, False, tunits = "sec", window = window)
    # mx = ufft.bandSumCentredOnPeaks(f, mx, 3)

    fft_utils.plotArray("FFT", "freq (Hz)", "Amplit", f, mx)
    tph = (1.0 / f[1:])
    fft_utils.plotArray("FFT", "Period (sec)", "Amplit", tph, mx[1:])

    # 3) confidence interval
    samp_rate = 20
    sim_time = 60
    nsamps = samp_rate * sim_time
    cuttoff_freq = 0.5

    fig = plt.figure()

    # generate input signal
    t = np.linspace(0, sim_time, nsamps)
    freqs = [0.1, 0.5, 1., 4.]
    x = 0
    for i in range(len(freqs)):
        x += np.cos(2 * math.pi * freqs[i] * t)
    plt.plot(t, x)

    filtered = True

    plt.title('Filter Input - Time Domain')
    plt.grid(True)

    # plain FFT no windowinf on the whole time series
    [y, Time, fftx, NumUniquePts, amplit, f, power] = fftsa.fourierTSAnalysis(t, x, False, tunits = "sec", window = window)
    dof = fft_utils.dof(f)  # one dof used for 'mean' the second lost due to association to the 'error'
    (x05, x95) = fft_utils.confidence_interval(amplit, dof, 0.95)
    fig_B = plt.figure()
    plt.plot(f, amplit, "k", f, x05, 'b', f, x95, 'r')
    plt.title('Confidence Interval')
    plt.legend(['series', '5%', '95%'])
    plt.ylabel("Amplitude")
    plt.grid(True)

    # TEST the SEGMANTING of the periodogram
    window = 'hanning'
    t1 = t[0:len(x) / 2]
    t2 = t[len(x) / 4:3 * len(x) / 4]
    t3 = t[len(x) / 2:len(x)]
    x1 = x[0:len(x) / 2]
    x2 = x[len(x) / 4:3 * len(x) / 4]
    x3 = x[len(x) / 2:len(x)]

    [y1, Time1, fftx1, NumUniquePts1, amplit1, f1, power1] = fftsa.fourierTSAnalysis(t1, x1, False, tunits = "sec", window = window)
    [y2, Time2, fftx2, NumUniquePts2, amplit2, f2, power2] = fftsa.fourierTSAnalysis(t2, x2, False, tunits = "sec", window = window)
    [y3, Time3, fftx3, NumUniquePts3, amplit3, f3, power3] = fftsa.fourierTSAnalysis(t3, x3, False, tunits = "sec", window = window)

    amplitude = (amplit1 + amplit2 + amplit3) / 3
    data_len = len(t)
    interval_len = len(x) / 2
    dof = fft_utils.edof(amplit, data_len, interval_len, window)  # one dt chunk see HArtman Notes ATM 552 page 159 example
    (x05, x95) = fft_utils.confidence_interval(amplit, dof, 0.95)
    fig_B = plt.figure()

    plt.plot(f, amplit, "k", f, x05, 'b', f, x95, 'r')
    plt.title('Confidence Interval Segmented data')
    plt.legend(['series', '5%', '95%'])
    plt.ylabel("Amplitude")
    plt.grid(True)

    fig_C = plt.figure()
    non_overlapping_seg = 8
    [freq, avg_fftx, avg_amplit, avg_power, x05, x95] = \
        fftsa.WelchFourierAnalysis_overlap50pct(t, x, False, tunits = "sec", window = window, nseg = non_overlapping_seg)
    plt.plot(freq, avg_amplit, "g", freq, x05, 'b', freq, x95, 'r')

    plt.title('Confidence Interval for Ovelapping Data Segments')
    plt.legend(['series', '5%', '95%'])
    plt.ylabel("Amplitude")
    plt.grid(True)


    plt.show()

