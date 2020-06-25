'''
Created on Sept 19, 2012

@author: Bogdan Hlevca
@email: bogdan@hlevca.com
@copyright:
    This module is based on Sebastian Krieger's kPyWavelet  email: sebastian@nublia.com

    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/, on
    routines provided by Aslak Grinsted, John Moore and Svetlana
    Jevrejeva and available at
    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
    on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

@version: 0.2
@date: October 1, 2014
@note: [1] Mallat, Stephane G. (1999). A wavelet tour of signal processing
       [2] Addison, Paul S. The illustrated wavelet transform handbook
       [3] Torrence, C. and Compo, G. P. (1998). A Practical Guide to
        Wavelet Analysis. Bulletin of the American Meteorological
        Society, American Meteorological Society, 1998, 79, 61-78.
       [4] Grinsted, A.; Moore, J. C. & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to
        geophysical time series. Nonlinear Processes in Geophysics,
        2004, 11, 561-566.

'''

import ufft.fft_utils as fft_utils
import pycwt.wav as wavelet
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

from datetime import datetime
from datetime import timedelta
from matplotlib.dates import seconds
from matplotlib.dates import date2num, num2date
from matplotlib.dates import MONDAY, SATURDAY
import matplotlib.dates


years = matplotlib.dates.YearLocator()  # every year
months = matplotlib.dates.MonthLocator()  # every month
yearsFmt = matplotlib.dates.DateFormatter('%Y')
# every monday
mondays = matplotlib.dates.WeekdayLocator(MONDAY)

class kCwt(object):
    '''
    Python class for wavelet analysis and the statistical approach
    suggested by Torrence and Compo (1998) using the wavelet module. To run
    this script successfully, the matplotlib module has to be installed

    IMPORTANT! Data input is in days since but it is converted in seconds for processing.
               All calculations are done in seconds and must be converted if we want something else

    '''
    class WavPar(object):



        def __init__(self, time, data, tunits):

            self.alpha = 0.0  # Lag-1 autocorrelation for white noise
            self.std = None  # Standard deviation
            self.variance = None  # Variance
            self.N = None  # timseries length
            self.freq = None
            self.period = None
            self.dt = None
            self.coi = None
            self.power = None  # Normalized wavelet power spectrum
            self.iwave = None  # inverse wavelet
            self.fft_power = None  # FFT power spectrum
            self.amplitude = None
            self.phase = None
            self.signal = None  # detrended timeseries
            self.glbl_power = None
            self.glbl_signif = None
            self.units = None
            self.fft = None
            self.fftfreqs = None
            self.scales = None
            self.wave = None
            self.tunits = None
            self.tfactor = 1
            self.sig95 = None

            self.data = data


            # the date from loggers is always transformed in days
            if time[1] < 695056:
                time += 695056
            if tunits == 'day':
                self.tfactor = 1
            elif tunits == 'hour':
                self.tfactor = 24
            elif tunits == 'sec':
                self.tfactor = 86400
            else:
                print("Wrong time units!")
                raise Exception('Error', 'Wrong time units!')
            # change to seconds after calculating tfactor
            if tunits != None:
                self.tunits = tunits
            else:
                self.tunits = 'day'

            print(">tunits :%s" % self.tunits)
            self.Time = time

    class XWavPar(object):
        '''
        XWT (array like) :
            Cross wavelet transform according to the selected mother
            wavelet.
                or
            Coherence wavelet transform according to the selected mother
            wavelet.
        x (array like) :
            Intersected independent variable.
        coi (array like) :
            Cone of influence, which is a vector of N points containing
            the maximum Fourier period of useful information at that
            particular time. Periods greater than those are subject to
            edge effects.
        freqs (array like) :
            Vector of Fourier equivalent frequencies (in 1 / time units)
            that correspond to the wavelet scales.
        signif (array like) :
            Significance levels as a function of scale.
        '''

        def __init__(self, xwt):
            self.XWT, self.x, self.coi, self.freqs, self.signif = xwt
            self.angle = None
            self.y1 = None
            self.y2 = None

    class CWavPar(object):
        '''
        WCT (array like) :
            Cross wavelet transform according to the selected mother
            wavelet.
                or
            Coherence wavelet transform according to the selected mother
            wavelet.
        x (array like) :
            Intersected independent variable.
        coi (array like) :
            Cone of influence, which is a vector of N points containing
            the maximum Fourier period of useful information at that
            particular time. Periods greater than those are subject to
            edge effects.
        freqs (array like) :
            Vector of Fourier equivalent frequencies (in 1 / time units)
            that correspond to the wavelet scales.
        signif (array like) :
            Significance levels as a function of scale.
        '''

        def __init__(self, xwt):
            self.XWT, self.x, self.coi, self.freqs, self.signif, self.angle = xwt
            self.y1 = None
            self.y2 = None



    # End WavPar


    def __init__(self, *args, **kwargs):

        self.wpar1 = None
        self.wpar2 = None
        self.wparrot = None
        self.wparwct = None
        self.wparxwt = None
        # Data members
        self.mother = None
        self.dj = 0.25  # Four sub-octaves per octaves
        self.s0 = -1  # 2 * dt                      # Starting scale, here 6 months
        self.J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
        self.cross = False
        self.rotary = False

        if len(args) == 3:
            self.initFromFile(args[0], args[1], args[2])
        elif len (args) == 4:
            self.initFromData(args[0], args[1], args[2], args[3])
        elif len (args) == 5:
            self.initFromDataRotary(args[0], args[1], args[2], args[3], args[4])
            self.rotary = True
        else :
            self.initFromDataCross(args[0], args[1], args[2], args[3], args[4], args[5])
            self.cross = True
            self.rotary = False

    def get_xwt(self):
        return self.wparxwt

    def get_wct(self):
        return self.wparwct

    def initFromData(self, Time, data, tunits = "day", readfile = False):
        self.wpar1 = self.WavPar(np.array(Time), np.array(data), tunits)

    def initFromDataRotary(self, Time, data, data1, tunits = "day", readfile = False):
        self.wpar2 = self.WavPar(np.array(Time), np.array(data1), tunits)
        self.initFromData(Time, data, tunits, readfile)

    def initFromDataCross(self, Time, data, Time1, data1, tunits = "day", readfile = False):
        self.wpar2 = self.WavPar(np.array(Time1), np.array(data1), tunits)
        self.initFromData(Time, data, tunits, readfile)

    def initFromFile(self, path, file, tunits = "day"):

        '''
        @param path: path to the data file
        @param file: file name of the data file
        @param tunits:"day", "hour", "sec" the unit of the time interval in the timeseries
        @param time: the time array, used for passing data directly, when not reading from a file
        @param var: the timeseries data, used for passing data directly, when not reading from a file

        '''
        self.filename = file
        # read Lake data
        [self.Time, self.data] = fft_utils.readFile(path, file)
        self.wpar1 = self.WavPar(np.array(self.Time), np.array(self.data), tunits)


    def doSpectralAnalysis(self, title, motherW = 'morlet', slevel = None, avg1 = None, avg2 = None, \
                           dj = None, s0 = None, J = None, alpha = None, counterclock = True) :
        '''
        Calls the required function in sequence to perform a wavelet and Fourier Analysis

        @param title: Title of the analysis
        @param motherW: the mother wavelet name
        @param slevel: the significant level abaove which the value can't be considered random. default:0.95
        @param avg1: First value in a range of Y axis type  to plot in scalogram (ex periods)
        @param avg2: Last in a range of Y axis type  to plot in scalogram (ex periods)

        @param dt: float Sample spacing.
        @param dj: (float, optional) : Spacing between discrete scales. Default value is 0.25.
                   Smaller values will result in better scale resolution, but
                   slower calculation and plot.
        @param s0: (float, optional) : Smallest scale of the wavelet. Default value is 2*dt.
        @param J: (float, optional) : Number of scales less one. Scales range from s0 up to
                   s0 * 2**(J * dj), which gives a total of (J + 1) scales.
                   Default is J = (log2(N*dt/so))/dj.

        @return: None

        '''

        # CHECK IF THIS IS NOT MISUSED
        if self.cross:
            raise Exception("Error: For cross wavelet use doCrossSpectralAnalysis() instead")

        self.dj = dj
        self.s0 = s0
        self.J = J
        self.alpha = alpha
        self.title = title
        self.avg1 = avg1
        self.avg2 = avg2
        if motherW == 'dog':
            self.mother = wavelet.DOG()
        elif motherW == 'morlet':
            self.mother = wavelet.Morlet(6.)

        # [self.Time, data1, X1, scales1, freq1, corr1 ]
        self._doSpectralAnalysisOnSeries(counterclock)

        if slevel != None:
            if self.rotary or self.cross:
                self.get95Significance(self.wpar2, slevel)
                self.getGlobalSpectrum(self.wpar2, slevel)
            # endif
            self.get95Significance(self.wpar1, slevel)
            self.getGlobalSpectrum(self.wpar1, slevel)

        else:
            print("Call get95Significance(slevel) & getGlobalSpectrum(slevel) manually")
        if avg1 != None and avg2 != None:
            self.getScaleAverageSignificance(slevel, self.wpar1, avg1, avg2)
            if self.rotary == True or self.cross == True:
                self.getScaleAverageSignificance(slevel, self.wpar2, avg1, avg2)
        else:
            print("Call getScaleAverageSignificance() manually")



    def doCrossSpectralAnalysis(self, title, motherW = 'morlet', slevel = None, avg1 = None, avg2 = None,
                           dj = None, s0 = None, J = None, alpha = None) :
        '''
        Calls the required function in sequence to perform a wavelet and Fourier Analysis

        @param title: Title of the analysis
        @param motherW: the mother wavelet name
        @param slevel: the significan level abaove which the value can't be considered random. default:0.95
        @param avg1: First value in a range of Y axis type  to plot in scalogram (ex periods)
        @param avg2: Last in a range of Y axis type  to plot in scalogram (ex periods)

        @param dt: float Sample spacing.
        @param dj: (float, optional) : Spacing between discrete scales. Default value is 0.25.
                   Smaller values will result in better scale resolution, but
                   slower calculation and plot.
        @param s0: (float, optional) : Smallest scale of the wavelet. Default value is 2*dt.
        @param J: (float, optional) : Number of scales less one. Scales range from s0 up to
                   s0 * 2**(J * dj), which gives a total of (J + 1) scales.
                   Default is J = (log2(N*dt/so))/dj.

        @return: None

        '''
        self.dj = dj
        self.s0 = s0
        self.J = J
        self.alpha = alpha
        self.slevel = slevel
        self.title = title
        self.avg1 = avg1
        self.avg2 = avg2

        if motherW == 'dog':
            self.mother = wavelet.DOG()
        elif motherW == 'morlet':
            self.mother = wavelet.Morlet(6.)

        self._doCrossSpectralAnalysisOnSeries()
        if slevel != None:
            self.get95Significance(self.wpar2, slevel)
            self.getGlobalSpectrum(self.wpar2, slevel)
            self.get95Significance(self.wpar1, slevel)
            self.getGlobalSpectrum(self.wpar1, slevel)
        else:
            print("Call get95Significance(slevel) & getGlobalSpectrum(slevel) manually")
        if avg1 != None and avg2 != None:
            self.getScaleAverageSignificance(slevel, self.wpar1, avg1, avg2)
            self.getScaleAverageSignificance(slevel, self.wpar2, avg1, avg2)
        else:
            print("Call getScaleAverageSignificance() manually")


    def _doSpectralAnalysisOnSeries(self, counterclock=True):
        ''' private workhorse
        '''
        # check if we have vector data (u and v components)
        if self.wpar2 is not None:
            self.Transform(self.wpar1)
            self.Transform(self.wpar2)
            [fftdata, fftpowerdata, wavedata, iwavedata, powerwavedata, amplitude, phase, std, coi, freq, scales, fftfreqs] = \
                self.rotary_spectra(self.wpar1, self.wpar2)
            [wpuv, wquv, wcw, wccw] = wavedata
            [iwpuv, iwquv, iwcw, iwccw] = iwavedata
            [pwpuv, pwquv, pwcw, pwccw] = powerwavedata
            [fpuv, fquv, fcw, fccw] = fftdata
            [pfpuv, pfquv, pfcw, pfccw] = fftpowerdata

            self.wparrot = self.WavPar(self.wpar1.Time, self.wpar1.data + 1j * self.wpar2.data, self.wpar1.tunits)

            self.wparrot.scales = scales
            self.wparrot.freq = freq
            self.wparrot.coi = coi
            self.wparrot.fftfreqs = fftfreqs
            self.wparrot.amplitude = amplitude
            self.wparrot.phase = phase
            self.wparrot.std = std
            self.wparrot.period = 1. / freq
            self.wparrot.variance = self.wparrot.std ** 2  # Variance

            if counterclock:
                self.wparrot.fft = fccw
                self.wparrot.fft_power = pfccw
                self.wparrot.iwave = iwccw
                self.wparrot.power = pwccw
                self.wparrot.wave = wccw
            else:
                self.wparrot.fft = fcw
                self.wparrot.fft_power = pfcw
                self.wparrot.iwave = iwcw
                self.wparrot.power = pwcw
                self.wparrot.wave = wcw
        else:
            print("Single data transformation")
            self.Transform(self.wpar1)

        # end if counter

    # end def _doSpectralAnalysisOnSeries


    def _doCrossSpectralAnalysisOnSeries(self):
        ''' private workhorse
        '''

        assert len(self.wpar1.data) == len(self.wpar2.data), "Error: data1 and data2 sizes differ. Their length must be the same"

        # check if we have vector data (u and v components)
        self.Transform(self.wpar1)
        self.Transform(self.wpar2)


        dt = (self.wpar1.Time[2] - self.wpar1.Time[1]) * self.wpar1.tfactor

        # Calculate the cross wavelet transform (XWT). The XWT finds regions in time
        # frequency space where the time series show high common power. Torrence and
        # Compo (1998) state that the percent point function -- PPF (inverse of the
        # cumulative distribution function) of a chi-square distribution at 95%
        # confidence and two degrees of freedom is Z2(95%)=3.999. However, calculating
        # the PPF using chi2.ppf gives Z2(95%)=5.991. To ensure similar significance
        # intervals as in Grinsted et al. (2004), one has to use confidence of 86.46%.
        xwt = wavelet.xwt(self.wpar1.Time, self.wpar1.data, self.wpar2.Time, self.wpar2.data, \
                          significance_level = 0.8646, normalize = True)

        self.wparxwt = self.XWavPar(xwt)

        # Calculate the wavelet coherence (WTC). The WTC finds regions in time
        # frequency space where the two time seris co-vary, but do not necessarily have
        # high power.
        wct = wavelet.wct(self.wpar1.Time, self.wpar1.data, self.wpar2.Time, self.wpar2.data, \
                          significance_level = 0.8646, normalize = True)
        self.wparwct = self.CWavPar(wct)

    # end def _doCrossSpectralAnalysisOnSeries



    def Transform(self, wpar):

        if self.dj == None:
            self.dj = 0.25  # Four sub-octaves per octaves

        if self.s0 == None:
            self.s0 = -1  # 2 * dt                      # Starting scale, here 6 months

        if self.J == None:
            self.J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves

        if wpar.alpha == None:
            wpar.alpha = 0.0  # Lag-1 autocorrelation for white noise

        # 'dt' is a time step in the time series
        dataY = mlab.detrend_linear(wpar.data)


        wpar.dt = (wpar.Time[1] - wpar.Time[0])
        wpar.std = np.std(wpar.data)  # Standard deviation
        wpar.variance = wpar.std ** 2  # Variance

        # normalize by standard deviation (not necessary, but makes it easier
        # to compare with plot on Interactive Wavelet page, at
        # "http://paos.colorado.edu/research/wavelets/plot/"
        wpar.signal = (wpar.data - wpar.data.mean()) / wpar.std  # Calculating anomaly and normalizing

        # The following routines perform the wavelet transform and siginificance
        # analysis for the chosen data set.
        wpar.wave, wpar.scales, wpar.freq, wpar.coi, wpar.fft, wpar.fftfreqs = \
            wavelet.cwt(wpar.signal, wpar.dt, self.dj, self.s0, self.J, self.mother, None)

        # this should reconstruct the initial signal
        wpar.iwave = wavelet.icwt(wpar.wave, wpar.scales, wpar.dt, self.dj, self.mother)
        wpar.N = wpar.data.shape[0]

        # calculate power and amplitude spectrogram
        wpar.power = (np.abs(wpar.wave)) ** 2  # Normalized wavelet power spectrum
        wpar.fft_power = wpar.variance * np.abs(wpar.fft) ** 2  # FFT power spectrum
        wpar.amplitude = wpar.std * np.abs(wpar.wave) / 2.  # we use only half of the symmetrical
                                                                 # spectrum therefore divide by 2
        wpar.phase = np.angle(wpar.wave)

        wpar.period = 1. / wpar.freq

        return wpar
        # [wave, scales, freq, coi, ufft, fftfreqs, iwave, power, fft_power, amplitude, phase, std]


    def rotary_spectra_transforms(self, u, v):
        # autospectra of the scalar components
        pu = u * u.conj()
        pv = v * v.conj()

        # cross spectra
        puv = u.real * v.real + u.imag * v.imag

        # quadrature spectra
        quv = -u.real * v.imag + v.real * u.imag

        # rotatory components
        cw = (pu + pv - 2 * quv) / 8
        ccw = (pu + pv + 2 * quv) / 8
        return [puv, quv, cw, ccw]

    def rotary_spectra(self, wpar1, wpar2):
        '''
        Trans has the components:
               wave, scales, freq, coi, ufft, fftfreqs, iwave, power, fft_power, amplitude, phase
        '''

        # FFT
        fftdata = [fpuv, fquv, fcw, fccw] = self.rotary_spectra_transforms(wpar1.fft, wpar2.fft)
        fftpowerdata = [pfpuv, pfquv, pfcw, pfccw] = self.rotary_spectra_transforms(wpar1.fft_power, wpar2.fft_power)

        # wavelet
        wavedata = [wpuv, wquv, wcw, wccw] = self.rotary_spectra_transforms(wpar1.wave, wpar2.wave)
        iwavedata = [iwpuv, iwquv, iwcw, iwccw] = self.rotary_spectra_transforms(wpar1.iwave, wpar2.iwave)
        powerwavedata = [pwpuv, pwquv, pwcw, pwccw] = self.rotary_spectra_transforms(wpar1.power, wpar2.power)

        amplitude = wpar1.amplitude + wpar2.amplitude * 1j
        phase = wpar1.phase + wpar2.phase * 1j

        std = max(wpar1.std, wpar2.std)

        coi = min(np.min(wpar1.coi), np.min(wpar2.coi))  # the are the same if size of the vector is the same
        freq = wpar1.freq  # same as above
        scales = wpar1.scales
        fftfreqs = wpar1.fftfreqs

        return [fftdata, fftpowerdata, wavedata, iwavedata, powerwavedata, amplitude, phase, std, coi, freq, scales, fftfreqs]


    def get95Significance(self, wpar, slevel):
        signif, fft_theor = wavelet.significance(1.0, wpar.dt, wpar.scales, 0, wpar.alpha, \
                                                 significance_level = slevel, wavelet = self.mother)

        sig95 = (signif * np.ones((wpar.N, 1))).transpose()
        wpar.sig95 = wpar.power / sig95  # Where ratio > 1, power is significant
        return wpar.sig95

    def getGlobalSpectrum(self, wpar, slevel):
        wpar.glbl_power = wpar.variance * wpar.power.mean(axis = 1)
        dof = wpar.N - wpar.scales  # Correction for padding at edges
        wpar.glbl_signif, tmp = wavelet.significance(wpar.variance, wpar.dt, wpar.scales, 1, wpar.alpha, \
                                                significance_level = slevel, dof = dof, wavelet = self.mother)

        return wpar.glbl_signif



    def getScaleAverageSignificance(self, slevel, wpar, avg1, avg2):
        # Scale average between avg1 and avg2 periods and significance level
        sel = pylab.find((wpar.period >= avg1) & (wpar.period < avg2))
        Cdelta = self.mother.cdelta

        # ones: Return a new array of given shape and type, filled with ones.
        scale_avg = (wpar.scales * np.ones((wpar.N, 1))).transpose()  # expand scale --> (J+1)x(N) array
        scale_avg = wpar.power / scale_avg  # [Eqn(24) Torrence & Compo (1998)

        # Cdelta = shape factor depeding on the wavelet used.
        #
        # To examine fluctuations in power over a range of scales/frequencies one can define a
        # scale averaged wavelet power" as a weighted sum of power spectrum over scales s1 to s2
        # here defined by the selected between avg1 and avg2
        # ]
        # By comparing [24] with [Eq 14] it can be shown that self.scale_avg is the average variance in a certain band
        # Here W[n]^2/s[j] = self.power / scale_avg , when n is the time index
        # sum(axis=0)  = sum over the scales
        #
        # Note: This can be used to examine the modulation of one time series by another or modulation of one frequency
        #       by another within the same timeseries (pag 73 Terrence & Compo (1998))
        wpar.scale_avg = wpar.variance * self.dj * wpar.dt / Cdelta * scale_avg[sel, :].sum(axis = 0)  # [Eqn(24)]

        # calculate the significant level for the averaged scales to represent the 95% (slevel) confidence interval
        wpar.scale_avg_signif, tmp = wavelet.significance(wpar.variance, wpar.dt, wpar.scales, 2, wpar.alpha,
                            significance_level = slevel, dof = [wpar.scales[sel[0]],
                            wpar.scales[sel[-1]]], wavelet = self.mother)
        return wpar.scale_avg_signif


    def plotAmplitudeSpectrogram(self, wpar, ylabel_ts, units_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2):
        '''
         The following routines plot the results in three different figures:
         - the global wavelet spectrum,
         - the wavelet amplitude spectrum,
         - the wavelet phase spectrum
         and Fourier spectra and finally the range averaged wavelet spectrum. In all
         sub-plots the significance levels are either includesuggested by Torrence and Compo (1998) using the wavelet module.
         To run this script successfully, the matplotlib module has to be installedd as dotted lines or as
         filled contour lines.

         @param ylablel_ts: label on the y axis on the data plot a) - string
         @param units_ts: units name for Y axis
         @param xlabel_sc: label to be placed on the X axis om the scalogram b) - string
         @param ylabel_sc: label to be placed on the Y axis om the scalogram b) - string
         @param sx_type: 'period' or 'freq' - creates the y axis on scalogram as scales/period or frequency
         @param x_type: 'date' will format the X axis as a date, 'time' will use regular numbers for time sequence
         @param val1: Range of sc_type (ex periods) to plot in scalogram
         @param val2: Range of sc_type (ex periods) to plot in scalogram

         @return: None
        '''

        if x_type == 'dayofyear':
            Time = fft_utils.timestamp2doy(wpar.Time)
        else :
            Time = wpar.Time


        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1, 0.1, 0.7, 0.60])
        ax3 = fig1.add_axes([0.83, 0.1, 0.03, 0.6])
        ax1.set_yscale('log')
        im1 = ax1.pcolormesh(Time, wpar.freq, wpar.amplitude)

        fig2 = plt.figure()
        ax2 = fig2.add_axes([0.1, 0.1, 0.7, 0.60])
        ax4 = fig2.add_axes([0.83, 0.1, 0.03, 0.6])
        ax2.set_yscale('log')
        im2 = ax2.pcolormesh(Time, wpar.freq, wpar.phase)

        # set correct way of axis, whitespace before and after with window
        # length
        ax1.axis('tight')
        # ax.set_xlim(0, end)
        ax1.grid(False)
        ax1.set_xlabel(xlabel_sc)
        ax1.set_ylabel(ylabel_sc)
        ax1.set_title('Amplitude ' + self.title)
        fig1.colorbar(im1, cax = ax3)

        if x_type == 'date':
            formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(formatter)
            ax1.xaxis.set_minor_locator(mondays)
            ax1.xaxis.grid(True, 'minor')
            fig1.autofmt_xdate()

        # set correct way of axis, whitespace before and after with window
        # length
        ax2.axis('tight')
        # ax.set_xlim(0, end)
        ax2.grid(False)
        ax2.set_xlabel(xlabel_sc)
        ax2.set_ylabel(ylabel_sc)
        ax2.set_title('Phase - ' + self.title)
        fig2.colorbar(im2, cax = ax4)

        fig3 = plt.figure()
        plt.title('Global Wavelet Spectrum Amplitude')

        A = np.sqrt(wpar.glbl_power) / 2.
        plt.plot(wpar.freq, A)
        plt.grid(True)
        plt.show()


    def plotSpectrogram(self, wpar, ylabel_ts, units_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2, \
                        raw = False, title = False, powerimg = False, bcolbar = False):
        '''
         The following routines plot the results in four different subplots containing:
         - the original series,
         - the wavelet power spectrum,
         - the global wavelet and Fourier spectra
         - the range averaged wavelet spectrum.
         In all sub-plots the significance levels are either includesuggested by Torrence and Compo (1998)
         using the wavelet module.

          :param ylablel_ts: label on the y axis on the data plot a) - string
          :param units_ts: units name for Y axis
          :param xlabel_sc: label to be placed on the X axis om the scalogram b) - string
          :param ylabel_sc: label to be placed on the Y axis om the scalogram b) - string
          :param sx_type: 'period' or 'freq' - creates the y axis on scalogram as scales/period or frequency
          :param x_type: 'date' will format the X axis as a date, 'time' will use regular numbers for time sequence
          :param val1: Range of sc_type (ex periods) to plot in scalogram
          :param val2: Range of sc_type (ex periods) to plot in scalogram
          :param raw: True/False,
          :param title : True/False -- whether to prin the title
          :param powerimg: True/False -- whether to display the power images instead of wavelet  (only for roatery)
          :param bcolbar: True/False - whether to draw the color bar
          @return: None

        '''
        fontsize = 14
        pylab.close('all')
        # fontsize = 'medium'
        params = {'font.size': fontsize - 2,
                  'xtick.labelsize': fontsize - 2,
                  'ytick.labelsize': fontsize - 2,
                  'axes.titlesize': fontsize,
                  'axes.labelsize': fontsize,
                  'text.usetex': True
                 }
        pylab.rcParams.update(params)  # Plot parameters
        figprops = dict(figsize = (11, 8), dpi = 96)
        fig = plt.figure(**figprops)
        units = units_ts
        # First sub-plot, the original time series anomaly.


        if x_type == 'dayofyear':
            Time = fft_utils.timestamp2doy(wpar.Time)
        elif x_type == 'date':
            Time = wpar.Time
        else:
            Time = wpar.Time

        if raw :
            ax = fig.add_axes([0.1, 0.75, 0.64, 0.2])
            # Plot the reconstructed signal. They are close to the original in case of simple signals.
            # The longer and more complex the signal is the more difficult is to cecomstruct.
            # The reconstructed signal is usually symmetrical
            ax.plot(Time, wpar.iwave, '-', linewidth = 1, color = [0.5, 0.5, 0.5])

            # Plot the original signal
            ax.plot(Time, wpar.data, 'k', linewidth = 1.5)

            if title:
                ax.set_title('(a) %s' % (self.title,))

            if self.units != '':
              ax.set_ylabel(r'%s [$%s$]' % (ylabel_ts, units,))
            else:
              ax.set_ylabel(r'%s' % (ylabel_ts,))

        # Second sub-plot, the normalized wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.
        if raw:
            bx = fig.add_axes([0.1, 0.37, 0.64, 0.28], sharex = ax)
        else :
            bx = fig.add_axes([0.1, 0.55, 0.64, 0.38])

        if x_type == 'date':
            formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
            if raw:
                axis = ax
            else :
                axis = bx

            axis.xaxis.set_major_formatter(formatter)
            axis.xaxis.set_minor_locator(mondays)
            axis.xaxis.grid(True, 'minor')

            fig.autofmt_xdate()



        # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        if sc_type == 'freq':
            y_scales = wpar.freq
        else:
            y_scales = wpar.period
            # levels = np.arange(np.log2(self.period.min()), np.log2(self.period.max()), (np.log2(self.period.max()) - np.log2(self.period.min())) / 10)

        sel = pylab.find((y_scales >= val1) & (y_scales < val2))  # indices of selected data sales or freq
        y_scales = y_scales[sel[0]:sel[len(sel) - 1] + 1]
        power = wpar.power[sel[0]:sel[len(sel) - 1] + 1]
        wave = wpar.wave[sel[0]:sel[len(sel) - 1] + 1]
        sig95 = wpar.sig95[sel[0]:sel[len(sel) - 1] + 1]
        glbl_signif = wpar.glbl_signif[sel[0]:sel[len(sel) - 1] + 1]
        glbl_power = wpar.glbl_power[sel[0]:sel[len(sel) - 1] + 1]

        # levels = np.arange(power.min(), power.max() + power.min(), (power.max() - power.min()) / 32)
        if self.rotary:
            if powerimg:
                lev_exp = np.arange(np.floor(np.log2(power.min()) - 1), np.ceil(np.log2(power.max()) + 1))
                levels = np.power(2, lev_exp)
                # im = bx.contourf(Time, np.log2(y_scales), np.log2(power), np.log2(levels), cmap = cm.jet, extend = 'both')
                im = bx.contourf(Time, np.log2(y_scales), np.log2(power), cmap = cm.jet, extend = 'both')
            else:
                # lev_exp = np.arange(np.floor(np.log2(wave.min()) - 1), np.ceil(np.log2(wave.max()) + 1), (np.floor(np.log2(wave.min()) - 1) - np.ceil(np.log2(wave.max()) + 1)) / 25)
                # levels = np.power(2, lev_exp)

                im = bx.contourf(Time, np.log2(y_scales), np.log2(wave), 32, cmap = cm.jet, extend = 'both')
        else:
            if powerimg:
                im = bx.contourf(Time, np.log2(y_scales), np.log2(power), 32, cmap = cm.jet, extend = 'both')
            else:
                im = bx.contourf(Time, np.log2(y_scales), np.log2(wave), 32, cmap = cm.jet, extend = 'both')

        # For out of levels representation enable the following two lines.
        # However, the above lines need to define the required levels
        # im.cmap.set_under('yellow')
        # im.cmap.set_over('cyan')

        if not self.rotary:
            bx.contour(Time, np.log2(y_scales), sig95, [-99, 1], colors = 'k', linewidths = 1.7)

        bx.fill(np.concatenate([Time[:1] - wpar.dt, Time, Time[-1:] + wpar.dt, \
                                Time[-1:] + wpar.dt, Time[:1] - wpar.dt, Time[:1] - wpar.dt]), \
                                np.log2(np.concatenate([[1e-9], wpar.coi, [1e-9], y_scales[-1:], y_scales[-1:], [1e-9]]))\
                                , 'k', alpha = 0.3, hatch = 'x')



        # for testing only
        if bcolbar:
            fig.colorbar(im)  # - if present it will shift the scales
        if title:
            if raw:
                bx.set_title('(b) Wavelet Power Spectrum (%s)' % (self.mother.name))
            else:
                bx.set_title('(a) Wavelet Power Spectrum (%s) - %s' % (self.mother.name, self.title))
        # end if
        bx.set_ylabel(ylabel_sc)
        # Yticks = np.arange(y_scales.min(), y_scales.max(), (y_scales.max() - y_scales.min()) / 16)
        # formatter = FormatStrFormatter('%2.4f')
        # bx.yaxis.set_major_formatter(formatter)

        Yticks = 2 ** np.arange(np.ceil(np.log2(y_scales.min())), np.ceil(np.log2(y_scales.max())))
        bx.set_yticks(np.log2(Yticks))
        # time is in days , convert to hours or seconds just for scales
        print("Change y ticks label")
        ytl = bx.get_yticks().tolist()
        k = 0
        for tk in Yticks:
            ytl[k] = '%.2f' % (tk * wpar.tfactor)
            k += 1
        bx.set_yticklabels(ytl)
        print(">> ylabels", ytl)
        # labels = bx.get_yticklabels()
        # for t in labels: print t
        # pylab.setp(labels, visible = True)

        # bx.set_yticklabels(Yticks)

        # formatter = FuncFormatter(self.scinot)
        # bx.yaxis.set_major_formatter(formatter)
        bx.invert_yaxis()
        if x_type == 'date':
            bx.xaxis.set_major_formatter(formatter)
            bx.xaxis.set_minor_locator(mondays)
            bx.xaxis.grid(True, 'minor')
            fig.autofmt_xdate()

        bx.set_xlim([Time.min(), Time.max()])

        # Third sub-plot, the global wavelet and Fourier power spectra and  kwavelet.tunitstheoretical
        # noise spectra.

        if raw:
            cx = fig.add_axes([0.78, 0.37, 0.19, 0.28], sharey = bx)
        else :
            cx = fig.add_axes([0.78, 0.55, 0.19, 0.38], sharey = bx)

        # plot the Fourier power spectrum first
        cx.plot(wpar.fft_power, np.log2(1. / wpar.fftfreqs), '-', color = [0.6, 0.6, 0.6], linewidth = 1.)

        # plot the wavelet global ower
        cx.plot(glbl_power, np.log2(y_scales), 'k-', linewidth = 1.5)

        # the line of chosen significance, ususaly 95%
        cx.plot(glbl_signif, np.log2(y_scales), 'k-.')

        if title:
            if raw:
                cx.set_title('(c) Global Wavelet Spectrum')
            else:
                cx.set_title('(b) Global Wavelet Spectrum')

        if units != '':
          cx.set_xlabel(r'Power $[%s]^2$' % (units,))
        else:
          cx.set_xlabel(r'Power')

        cx.set_xlim([0, glbl_power.max() + wpar.variance])
        cx.set_ylim(np.log2([y_scales.min(), y_scales.max()]))

        # not necessary yaxis is shared
        # cx.set_yticks(np.log2(Yticks))
        # cx.set_yticklabels(Yticks)

        Xticks = np.arange(0, glbl_power.max() + glbl_power.max() / 4, glbl_power.max() / 3)
        cx.set_xticks(Xticks)
        if self.rotary:
            xlabels = ['%.3f' % i for i in Xticks]
        else:
            xlabels = ['%.3f' % i for i in Xticks]
        cx.set_xticklabels(xlabels)


        # cx.yaxis.set_major_formatter(formatter)

        pylab.setp(cx.get_yticklabels(), visible = False)
        cx.invert_yaxis()

        # Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
        # avg1 and avg2 parameters

        if raw :
            dx = fig.add_axes([0.1, 0.07, 0.64, 0.2], sharex = ax)
        else :
            dx = fig.add_axes([0.1, 0.07, 0.64, 0.3], sharex = bx)
        dx.axhline(wpar.scale_avg_signif, color = 'k', linestyle = '--', linewidth = 1.)

        # plot the scale average for each time point.
        dx.plot(Time, wpar.scale_avg, 'k-', linewidth = 1.5)

        if title:
            if raw:
                dx.set_title('(d) Scale-averaged power  [$%.4f$-$%.4f$] (%s)' % (self.avg1, self.avg2, wpar.tunits))
            else:
                dx.set_title('(c) Scale-averaged power  [$%.4f$-$%.4f$] (%s)' % (self.avg1, self.avg2, wpar.tunits))
        # ENDIF TITLE

        if x_type == 'dayofyear' :
            xlabel = 'Day of year'
        elif x_type == 'date':
            xlabel = 'Time (days)'
        else:
            xlabel = 'Time (%s)' % wpar.tunits
        dx.set_xlabel(xlabel)
        if units != '':
          dx.set_ylabel(r'Average variance [$%s$]' % (units,))
        else:
          dx.set_ylabel(r'Average variance')
        #
        if x_type == 'date':
            dx.xaxis.set_major_formatter(formatter)
            dx.xaxis.set_minor_locator(mondays)
            dx.xaxis.grid(True, 'minor')
            fig.autofmt_xdate()

        dx.set_xlim([Time.min(), Time.max()])
        #

        # pylab.draw()
        pylab.show()

    # Utility Functions used in the module
    def scinot(self, x, pos = None):
        '''
        Function to be used in the FuncFormatter to format scientific notation
        '''
        if x == 0:
            s = '0'
        else:
            xp = int(np.floor(np.log10(np.abs(x))))

            mn = x / 10.**xp
            # Here we truncate to 3 significant digits -- may not be enough
            # in all cases
            s = '$' + str('%.2f' % mn) + '\\times 10^{' + str(xp) + '}$'
            return s

    def boxpdf(self, x):
        """
        Forces the probability density function of the input data to have
        a boxed distribution.

        PARAMETERS
            x (array like) :
                Input data

        RETURNS
            X (array like) :
                Boxed data varying between zero and one.
            Bx, By (array like) :
                Data lookup table

        """
        x = np.asarray(x)
        n = x.size

        # Kind of 'unique'
        i = np.argsort(x)
        d = (np.diff(x[i]) != 0)
        I = pylab.find(np.concatenate([d, [True]]))
        X = x[i][I]

        I = np.concatenate([[0], I + 1])
        Y = 0.5 * (I[0:-1] + I[1:]) / n
        bX = np.interp(x, X, Y)

        return bX, X, Y


    def plotXSpectrogram(*args, **kwargs):
        """Plots the cross wavelet power spectrum and phase arrows.
        function.

        The relative phase relationship convention is the same as adopted
        by Torrence and Webster (1999), where in phase signals point
        upwards (N), anti-phase signals point downwards (S). If X leads Y,
        arrows point to the right (E) and if X lags Y, arrow points to the
        left (W).

        PARAMETERS
            xwt (array like) :
                Cross wavelet transform.
            coi (array like) :
                Cone of influence, which is a vector of N points containing
                the maximum Fourier period of useful information at that
                particular time. Periods greater than those are subject to
                edge effects.
            freqs (array like) :
                Vector of Fourier equivalent frequencies (in 1 / time units)
                that correspond to the wavelet scales.
            signif (array like) :
                Significance levels as a function of Fourier equivalent
                frequencies.
            da (list, optional) :
                Pair of integers that the define frequency of arrows in
                frequency and time, default is da = [3, 3]. 
                First digit controls the number of of arrows the Y axis; increased number ex 6  => more rows
                Second digit controls the number of of arrows the X axis; decreased number ex 1  => more columns
                
            tfactor (oprional) :
                the conversin factor bertween the real scales and the unit that needs to be displayed

        RETURNS
            A list with the figure and axis objects for the plot.

        SEE ALSO
            wavelet.xwt

        """
        fontsize = 18
        pylab.close('all')
        # fontsize = 'medium'
        params = {'text.fontsize': fontsize - 2,
                  'xtick.labelsize': fontsize,
                  'ytick.labelsize': fontsize - 2,
                  'axes.titlesize': fontsize + 2,
                  'axes.labelsize': fontsize + 2 ,
                  'text.usetex': True
                 }
        pylab.rcParams.update(params)  # Plot parameters


        # Sets some parameters and renames some of the input variables.
        xwavt = args[1]
        xwt = xwavt.XWT
        t = xwavt.x
        coi = xwavt.coi
        freqs = xwavt.freqs
        signif = xwavt.signif

        if 'scale' in list(kwargs.keys()):
            scale = kwargs['scale']
        else:
            scale = 'log2'

        N = len(t)
        period = 1. / freqs

        power = abs(xwt)
        sig95 = np.ones([1, N]) * signif[:, None]
        sig95 = power / sig95  # power is significant where ratio > 1

        # Calculates the phase between both time series. The phase arrows in the
        # cross wavelet power spectrum rotate clockwise with 'north' origin.
        if 'angle' in list(kwargs.keys()):
            angle = 0.5 * np.pi - kwargs['angle']
        else:
            angle = 0.5 * np.pi - np.angle(xwt)
        u, v = np.cos(angle), np.sin(angle)

        if 'da' in list(kwargs.keys()):
            da = kwargs['da']
            print(">> da :", da)
        else:
            da = [3, 3]

        if 'tfactor' in list(kwargs.keys()):
            tfactor = kwargs['tfactor']
        else:
            tfactor = 1
        figprops = dict(figsize = (11, 8), dpi = 96)
        fig = plt.figure(**figprops)
        ax = fig.add_subplot(1, 1, 1)

        # Plots the cross wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.
        if 'crange' in list(kwargs.keys()):
            levels = labels = kwargs['crange']
        else:
            levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
            labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']
        cmin, cmax = power.min(), power.max()
        rmin, rmax = min(levels), max(levels)
        if 'extend' in list(kwargs.keys()):
            extend = kwargs['extend']
        elif (cmin < rmin) & (cmax > rmax):
            extend = 'both'
        elif (cmin < rmin) & (cmax <= rmax):
            extend = 'min'
        elif (cmin >= rmin) & (cmax > rmax):
            extend = 'max'
        elif (cmin >= rmin) & (cmax <= rmax):
            extend = 'neither'

        if scale == 'log2':
            Power = np.log2(power)
            Levels = np.log2(levels)
        else:
            Power = power
            Levels = levels

        if 'x_type' in list(kwargs.keys()):
            x_type = kwargs['x_type']

            if x_type == 'dayofyear':
                Time = fft_utils.timestamp2doy(t)
                xlabel = 'Day of year'
                #xlabel = 'Julian Day'
            elif x_type == 'date':
                Time = t
                xlabel = 'Time (days)'
                ax.xaxis.set_major_formatter(formatter)
                ax.xaxis.set_minor_locator(mondays)
                ax.xaxis.grid(True, 'minor')
                fig.autofmt_xdate()
            # endif
            ax.set_xlabel(xlabel)

        else:
            Time = t
        # endif
        dt = (Time[1] - Time[0])

        if 'ylabel_sc' in list(kwargs.keys()):
            ylabel = kwargs['ylabel_sc']
        else:
            ylabel = 'Period'
        ax.set_ylabel(ylabel)


        cf = ax.contourf(Time, np.log2(period), Power, Levels, extend = extend)
        ax.contour(Time, np.log2(period), sig95, [-99, 1], colors = 'k',
            linewidths = 1.6)
        q = ax.quiver(Time[::da[1]], np.log2(period)[::da[0]], u[::da[0], ::da[1]],
            v[::da[0], ::da[1]], units = 'width', angles = 'uv', pivot = 'mid',
            linewidth = 0.6, edgecolor = 'k', headwidth = 5, headlength = 5,
            headaxislength = 3, minshaft = 3, minlength = 3)
        ax.fill(np.concatenate([Time[:1] - dt, Time, Time[-1:] + dt, Time[-1:] + dt, t[:1] - dt,
            t[:1] - dt]), np.log2(np.concatenate([[1e-9], coi, [1e-9],
            period[-1:], period[-1:], [1e-9]])), 'k', alpha = 0.3, hatch = 'x')
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(Yticks))
        # ax.set_yticklabels(Yticks)
        print("Change y ticks label")
        ytl = ax.get_yticks().tolist()
        k = 0
        for tk in Yticks:
            ytl[k] = '%.1f' % (tk * tfactor)
            k += 1
        ax.set_yticklabels(ytl)
        print(">> ylabels", ytl)

        ax.set_xlim([Time.min(), Time.max()])
        ax.set_ylim(np.log2([period.min(), min([coi.max(), period.max()])]))
        ax.invert_yaxis()
        cbar = fig.colorbar(cf, ticks = Levels, extend = extend)
        cbar.ax.set_yticklabels(labels)


        plt.show()

        return fig, ax


# test class
if __name__ == '__main__':
    '''
    Testing ground for local functions
    '''

    # 1) Test true amplitude
    Fs = 1000.0  # Sampling frequency
    T = 1.0 / Fs  # Sample time
    L = 1024  # Length of signal
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

    avg1, avg2 = (0.001, 0.03)  # Range of periods to average
    slevel = 0.95  # Significance level
    tunits = 'sec'
    # tunits = '^{\circ}C'
    kwavelet = kCwt(t, x, tunits, False)

    dj = 0.025  # Four sub-octaves per octaves
    s0 = -1  # 2 * dt                      # Starting scale, here 6 months
    J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
    alpha = 0.0  # Lag-1 autocorrelation for white noise
    kwavelet.doSpectralAnalysis(title, "morlet", slevel, avg1, avg2, dj, s0, J, alpha)
    ylabel_ts = "amplitude"
    yunits_ts = 'mm'
    xlabel_sc = ""
    ylabel_sc = 'Period (%s)' % kwavelet.tunits
    # ylabel_sc = 'Freq (Hz)'
    sc_type = "period"
    # sc_type = "freq"
    val1, val2 = (0.001, 0.02)  # Range of sc_type (ex periods) to plot in spectogram
    x_type = 'time'
    kwavelet.plotSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)
    kwavelet.plotAmplitudeSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)




