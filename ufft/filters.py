import scipy as sp
import numpy as np
from scipy.signal import freqz
import math
import ufft

def butterworth(data, btype, lowcut, highcut, fs, output = 'ba', passatten = 10, stopatten = 30, order = None, recurse = False, orders = None, recdepth = 0, debug = False):
    '''
     This function does butterworth filtering.

    Args:
       data (array):  the timeseries to be filtered.
       passband (float): the value from which we start filtering.
       stopband (float): the value from which we start filtering.
       btype  (string):  * 'low'  for low pass
                         * 'high' for high pass
                         * 'band' for band pass
       debug (Boolean): True/False - optional

    Kwargs:


    Returns:
       y (ndarray) : the filtered timeseries
       w (ndarray) : The frequencies at which h was computed.
       h (ndarray) : The frequency response.


    Raises:


    '''
    # design filter
    '''
    Lowpass    Wp < Ws, both scalars                          (Ws,1)                   (0,Wp)
    Highpass   Wp > Ws, both scalars                          (0,Ws)                   (Wp,1)
    Bandpass   The interval specified by Ws contains
               the one specified by Wp                        (0,Ws(1)) and (Ws(2),1)  (Wp(1),Wp(2))
               (Ws(1) < Wp(1) < Wp(2) < Ws(2)).

    Bandstop   The interval specified by Wp contains          (Ws(1),Ws(2))            (0,Wp(1)) and (Wp(2),1)
               the one specified by Ws
               (Wp(1) < Ws(1) < Ws(2) < Wp(2)).

    '''
    nqf = fs / 2.0
    ws = [lowcut / nqf, highcut / nqf]
    wp = [lowcut * 0.90 / nqf, highcut * 1.1 / nqf]


    if order == None:
        (N, Wn) = sp.signal.buttord(wp = wp, ws = ws, gpass = passatten, gstop = stopatten, analog = 0)
        if N > 8 : N = 6
    else:
        N = order
        Wn = wp

    width = (Wn[1] * nqf - Wn[0] * nqf)
    ratio = nqf / width
    if nqf / width > 200 and recdepth == 0 :  # decimate
        data = sp.signal.decimate(data, int(ratio / 200), n = 5, ftype = 'iir')
        # data = sp.signal.resample(data, len(data) / int(ratio / 200))

        fs = fs / int(ratio / 200)
        nqf = fs / 2.0
        ws = [lowcut / nqf, highcut / nqf]
        wp = [lowcut * 0.90 / nqf, highcut * 1.1 / nqf]
        Wn = wp
        width = (Wn[1] * nqf - Wn[0] * nqf)


    if recurse:
        if  recdepth == 0:
            step = 2
            orders = [  step if N - int(step * i / step) > step  else N - step * i / step   for i in range(0, N, step) ]
            if len(orders) > 0:
                N = orders[0]
            else:
                recurse = False

    if debug:
        print(" ============= DEBUG ============ ")
        print("cuttoff [%f - %f] hours" % ((1 / (Wn[0] * nqf)) / 3600, (1 / (Wn[1] * nqf)) / 3600))
        print("order = %d" % N)
        print("nyq = %f" % nqf)
        print("filter width: %f" % (width))


    if N > 8:
        print("Order of filter:%d too big to be stable, change parameters" % N)
        # N = 8
    elif N < 2:
        print("Order of filter:%d too small to be precise, change parameters" % N)
        # N = 2



    # If some values of b are too close to 0, they are removed. In that case, a :
    # BadCoefficients warning is emitted.
    if output == 'zpk':
        (z, p, k) = sp.signal.butter(N, Wn, btype = btype, analog = 0, output = 'zpk')

        if debug:
            print("Printing zeros:\n===============")
            for i in range(0, len(z)):
                print('z[%d]=' % i + ' ({0.real:.3f} + {0.imag:.7f}j)'.format(z[i]))
            print("Printing poles:\n===============")
            for i in range(0, len(p)):
                print('p[%d]=' % i + ' ({0.real:.3f} + {0.imag:.7f}j)'.format(p[i]))

            print("Printing Gain\n===============")
            print("k=%f" % (k))
        b, a = sp.signal.zpk2tf(z, p, k)

    elif output == 'ba':
        (b, a) = sp.signal.butter(N, Wn, btype = btype, analog = 0, output = 'ba')
        (z, p, k) = sp.signal.tf2zpk(b, a)
    else:
        err = "Butterworth: Unknown output type %s" % output
        raise BaseException(err)

    if debug == True:
        print(("b=" + str(b) + ", a=" + str(a)))

    # filter frequency response
    # w is in rad/sample
    (w, h) = sp.signal.freqz(b, a)

    # filtered output
    # zi = signal.lfiltic(b, a, x[0:5], x[0:5])
    # (y, zi) = signal.lfilter(b, a, x, zi=zi)
    y = sp.signal.filtfilt(b, a, data)
    delay = 0

    if recurse:
        depth = recdepth + 1
        if depth < len(orders):
            (y, w, h, N, delay) = butterworth(data, btype, lowcut, highcut, fs, output, passatten, stopatten, order = orders[depth], recurse = recurse, orders = orders, recdepth = depth, debug = debug)

    return (y, w, h, N, delay)


def chebyshev2(data, btype, lowcut, highcut, fs, output = 'ba', debug = False):
    '''
     This function does butterworth filtering.

    Args:
       data (array):  the timeseries to be filtered.
       passband (float): the value from which we start filtering.
       stopband (float): the value from which we start filtering.
       btype  (string):  * 'low'  for low pass
                         * 'high' for high pass
                         * 'band' for band pass
       debug (Boolean): True/False - optional

    Kwargs:


    Returns:
       y (ndarray) : the filtered timeseries
       w (ndarray) : The frequencies at which h was computed.
       h (ndarray) : The frequency response.


    Raises:


    '''
    # design filter
    '''
    Lowpass    Wp < Ws, both scalars                          (Ws,1)                   (0,Wp)
    Highpass   Wp > Ws, both scalars                          (0,Ws)                   (Wp,1)
    Bandpass   The interval specified by Ws contains
               the one specified by Wp                        (0,Ws(1)) and (Ws(2),1)  (Wp(1),Wp(2))
               (Ws(1) < Wp(1) < Wp(2) < Ws(2)).

    Bandstop   The interval specified by Wp contains          (Ws(1),Ws(2))            (0,Wp(1)) and (Wp(2),1)
               the one specified by Ws
               (Wp(1) < Ws(1) < Ws(2) < Wp(2)).

    '''
    nqf = fs / 2.0
    ws = [lowcut / nqf, highcut / nqf]
    wp = [lowcut * 0.90 / nqf, highcut * 1.1 / nqf]
    rs = 3.0
    (N, Wn) = sp.signal.cheb2ord(wp = wp, ws = ws, gpass = 7, gstop = 30, analog = 0)

    if output == 'zpk':
        (z, p, k) = sp.signal.cheby2(N, rs, Wn, btype = btype, analog = 0, output = 'zpk')
        b, a = sp.signal.zpk2tf(z, p, k)
    elif output == 'ba':
        (b, a) = sp.signal.cheby2(N, rs, Wn, btype = btype, analog = 0, output = 'ba')
    else:
        err = "Butterworth: Unknown output type %s" % output
        raise BaseException(err)


    # b *= 1e3
    if debug == True:
        print(("b=" + str(b) + ", a=" + str(a)))

    # filter frequency response
    # w is in rad/sample
    (w, h) = sp.signal.freqz(b, a)

    # filtered output
    # zi = signal.lfiltic(b, a, x[0:5], x[0:5])
    # (y, zi) = signal.lfilter(b, a, x, zi=zi)
    y = sp.signal.filtfilt(b, a, data)
    delay = 0
    return (y, w, h, N, delay)

def firwin(data, btype, lowcut, highcut, sample_rate, output = 'ba', debug = False):
    '''

    '''
    # design filter
    '''
    Lowpass    Wp < Ws, both scalars                          (Ws,1)                   (0,Wp)
    Highpass   Wp > Ws, both scalars                          (0,Ws)                   (Wp,1)
    Bandpass   The interval specified by Ws contains
               the one specified by Wp                        (0,Ws(1)) and (Ws(2),1)  (Wp(1),Wp(2))
               (Ws(1) < Wp(1) < Wp(2) < Ws(2)).

    Bandstop   The interval specified by Wp contains          (Ws(1),Ws(2))            (0,Wp(1)) and (Wp(2),1)
               the one specified by Ws
               (Wp(1) < Ws(1) < Ws(2) < Wp(2)).

    '''

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0


    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 * Nyq_rate in Hz transition width.
    width = (highcut - lowcut) / 2 / nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = sp.signal.kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    # cutoff_hz = highcut  # / 50.0
    cutoff_hz = [lowcut * 0.9, lowcut, highcut, highcut * 1.1]  # / 50.0

    # bands : array_like: A monotonic sequence containing the band edges in Hz. All elements must be non-negative and less than half the sampling frequency as given by Hz.
    # bands = np.array([0., lowcut * 0.90, lowcut, highcut, highcut * 1.1, nqf * 0.99])
    #                      stopband1     tranzition band1   passband   tranzition band2  stopband2

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    cutoff = np.divide(np.array(cutoff_hz), np.array([nyq_rate]))

    if N % 2 == 0: N += 1

    # taps = sp.signal.firwin(N, cutoff, window = ('kaiser', beta), nyq = nyq_rate)
    taps = sp.signal.firwin(25, cutoff, window = ('hamming'), nyq = 1)

    # Use lfilter to filter x with the FIR filter.
    y = sp.signal.lfilter(taps, 1.0, data)

    # filter frequency response
    # w is in rad/sample
    (w, h) = sp.signal.freqz(taps, worN = 8000)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate / 24 / 3600

    return (y, w, h, N, delay)



def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.signal.butter(order, [low , high], btype = 'band', analog = 0, output = 'ba')
    return b, a

def butter_highpass(highcut, fs, order = 5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, high, btype = 'high', analog = 0, output = 'ba')
    return b, a

def butter_lowpass(lowcut, fs, order = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, low, btype = 'low', analog = 0, output = 'ba')
    return b, a

#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
#    y = sp.signal.lfilter(b, a, data)
#    return y
#
# def butterworth_filter(data, btype = 'band', lowcut = None, highcut = None, fs = None, order = 5, worN = 2000):
#    if btype == 'band':
#        b, a = butter_bandpass(lowcut, highcut, fs, order = order)
#    elif btype == 'high':
#        b, a = butter_highpass(highcut, fs, order = order)
#    elif btype == 'low':
#        b, a = butter_lowpass(lowcut, fs, order = order)
#
#    w, h = sp.signal.freqz(b, a, worN = worN)
#    y = sp.signal.lfilter(b, a, data)
#
#    return [y, w, h, b, a]


def fft_bandpassfilter(data, fs, lowcut, highcut, force = None):
    '''
    This filter works for steady signals only - Do not use for oceanography
    '''
    print("This filter works for steady signals only - Do not use for oceanography!")

    if force == None:
        raise "This filter works for steady signals only - Do not use for oceanography! Use 'force = True to override"

    ufft = np.fft.fft(data)
    n = len(data)
    timestep = 1.0 / fs
    freq = np.fft.fftfreq(n, d = timestep)
    bp = ufft[:]
    for i in range(len(bp)):
        if freq[i] >= highcut or freq[i] < lowcut:
            bp[i] = 0
        #    print "Not Passed"
        # else :
        #    print "Passed"

    # must multipy by 2 to get the correct amplitude  due to FFT symetry
    ibp = sp.ifft(bp)
    return ibp

def fft_lowpass_filter(data, fs, lowcut, highcut):
    '''
    NOT properly tested
    Algo from: https://ccrma.stanford.edu/~jos/sasp/Example_1_Low_Pass_Filtering.html
    '''
    M = len(data)
    L = M + 1
    timestep = 1.0 / fs
    hsupp = np.arange(-(L - 1) / 2, (L - 1) / 2 + 1)
    hideal = (2 * lowcut / fs) * np.sinc(2 * lowcut * hsupp / fs)
    h = sp.signal.get_window('hamming', L).conj() * hideal
    Nfft = fft_utils.nextpow2(L + M - 1)
    xzp = np.concatenate((data, np.zeros(Nfft - M, dtype = np.float)))
    hzp = np.concatenate((h, np.zeros(Nfft - L, dtype = np.float)))
    X = np.fft.fft(xzp)  # signal
    H = np.fft.fft(hzp)  # filter
    Y = X * H
    ibp = sp.ifft(Y)

    return np.real(ibp)
