# from scipy.signal import butter, lfilter
import ufft.filters as filters

#===============================================================================
# def butter_bandpass(lowcut, highcut, fs, order = 5):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype = 'band')
#    return b, a
#
#
# def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
#    y = lfilter(b, a, data)
#    return y
#===============================================================================


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = filters.butter_bandpass(lowcut, highcut, fs, order = order)
        w, h = freqz(b, a, worN = 2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label = "order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label = 'sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc = 'best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint = False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label = 'Noisy signal')

    # y = filters.fft_lowpass_filter(x, lowcut, highcut, fs)
    y2, w, h, N, delay = filters.butterworth(x, 'pass', lowcut, highcut, fs, output = 'zpk', debug = 'False')
    # plt.plot(t, y[:len(t)], label = 'FFT Filtered signal (%g Hz)' % f0)
    plt.plot(t, y2, label = 'Butter Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles = '--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc = 'upper left')

    plt.show()

#
#    import scipy
#    from scipy import pi, sin, randn, fftpack
#    # from scipy.fftpack import *
#    from pylab import *
#    fs = 100.0
#    t = arange(0, 1, 1 / fs)
#    y = sin(2 * pi * t)
#    plot(t, y)
#    Y = scipy.fftpack.fftshift(fft(y, 256))
#    figure()
#    F = arange(-fs / 2, fs / 2, fs / len(Y))
#    plot(F, abs(Y))
#    show()
#
#    # Filtering example (plots just the frequency response)
#    from scipy import *
#    from scipy.signal import *
#    from pylab import *
#    b, a = butter(4, 0.5, btype = 'low', analog = 0)
#    w, h = freqz(b, a)
#    # plot(w,h);
#    b, a = butter(8, 0.8, btype = 'low')
#    w, h = freqz(b, a)
#    plot(w, h);
#    b, a = butter(16, 0.5, btype = 'low')
#    w, h = freqz(b, a)
#    # plot(w,h);
#    b, a = butter(32, 0.5, btype = 'low')
#    w, h = freqz(b, a)
#    plot(w, h);
#    # axis([])
#    hold(True)
#    h01 = firwin(8, 0.5, window = 'hamming')
#    h02 = firwin(8, 0.5, window = 'triang')
#    w, h01 = freqz(h01, 1)
#    plot(w, h01)
#    w, h02 = freqz(h02, 1)
#    plot(w, h02)
#    grid(True)
#    show()
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show


#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------

sample_rate = 100.0
nsamples = 400
t = arange(nsamples) / sample_rate
x = cos(2 * pi * 0.5 * t) + 0.2 * sin(2 * pi * 2.5 * t + 0.1) + \
        0.2 * sin(2 * pi * 15.3 * t) + 0.1 * sin(2 * pi * 16.7 * t + 0.1) + \
            0.1 * sin(2 * pi * 23.45 * t + .8)


#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0 / nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
cutoff_hz = 10.0

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz / nyq_rate, window = ('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)

#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1)
plot(taps, 'bo-', linewidth = 2)
title('Filter Coefficients (%d taps)' % N)
grid(True)

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(2)
clf()
w, h = freqz(taps, worN = 8000)
plot((w / pi) * nyq_rate, absolute(h), linewidth = 2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
ylim(-0.05, 1.05)
grid(True)

# Upper inset plot.
ax1 = axes([0.42, 0.6, .45, .25])
plot((w / pi) * nyq_rate, absolute(h), linewidth = 2)
xlim(0, 8.0)
ylim(0.9985, 1.001)
grid(True)

# Lower inset plot
ax2 = axes([0.42, 0.25, .45, .25])
plot((w / pi) * nyq_rate, absolute(h), linewidth = 2)
xlim(12.0, 20.0)
ylim(0.0, 0.0025)
grid(True)

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.
delay = 0.5 * (N - 1) / sample_rate

figure(3)
# Plot the original signal.
plot(t, x)
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t - delay, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth = 4)

xlabel('t')
grid(True)

show()
