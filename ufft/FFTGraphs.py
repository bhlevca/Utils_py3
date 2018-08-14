'''
Created on Jun 12, 2012

@author: bogdan
'''
import numpy as np
import scipy as sp

from . import fft_utils
from . import FFTSpectralAnalysis
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import warnings

# @staticmethod
def plotSingleSideAplitudeSpectrumFreqMultiple(lake_name, bay_names, data, freq, ci, num_segments = 1, \
                                               funits = "Hz", y_label = None, title = None, log = False, fontsize = 20, tunits = None):
    plotSingleSideSpectrumFreqMultiple(lake_name, bay_names, data, freq, ci, type = 'amplitude', num_segments = num_segments, \
                                               funits = funits, y_label = ylabel, title = title, log = log, fontsize = fontsize, tunits = tunits)


 # @staticmethod
def plotSingleSideSpectrumFreqMultiple(lake_name, bay_names, data, freq, ci = None, type = 'power', num_segments = 1, \
                                               funits = "Hz", y_label = None, title = None, log = False, fontsize = 20, tunits = None):

    # Plot single - sided amplitude spectrum.
    if title != None:
        title = title + " - Single-Sided Amplitude"

    if funits == 'Hz':
        xlabel = 'Frequency [Hz]'
    elif funits == 'cph':
        xlabel = 'Frequency [cph]'

    # end if

    if y_label == None:
        if type == "power":
            ylabel = "Spectral density [m$^2$/Hz]"
        else:
            ylabel = 'Z(t) [m]'
    else :
        ylabel = y_label

    legend = []
    ci05 = []
    ci95 = []
    XA = []
    YA = []
    for j in range(0, len(bay_names)):
        legend.append(bay_names[j])

         # smooth only if not segmented
        if num_segments == 1:
            sSeries = fft_utils.smoothSeries(data[j], 5)
        else:
            sSeries = data[j]
        # end
        if funits == 'cph':
            f = freq[j] * 3600
        else:
            f = freq[j]

        XA.append(f)
        YA.append(sSeries)
        if num_segments != 1 and ci != None:
            ci05.append(ci[0][j])
            ci95.append(ci[1][j])
            
    # end
    xa = np.array(XA)
    ya = np.array(YA)

    if num_segments == 1:
        fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, log)
    else:
        fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel, xa, ya, ci05, ci95, legend = legend, log = log, fontsize = fontsize)

# end plotSingleSideAplitudeSpectrumFreqMultiple

class FFTGraphs(object):
    '''
    classdocs
    '''

    def __init__(self, path = None, file1 = None, file2 = None, show = None, data = None, data1 = None, data2 = None):
        '''
        Constructor
        '''
        if path != None and file1 != None:
            self.path_in = path
            self.filename = file1
            self.filename1 = file2
            self.fftsa = FFTSpectralAnalysis.FFTSpectralAnalysis(path, file1, file2)
        elif data != None:
            self.path_in = None
            self.filename = None
            self.filename1 = None
            self.fftsa = FFTSpectralAnalysis.FFTSpectralAnalysis(None, None, None, data, data1)

        self.show = show
        self.mx = None  # Amplitude values from spectral analysis Lake
        self.mx1 = None  # Amplitude values from spectral analysis Lake
        self.mx2 = None  # Amplitude values from spectral analysis Lake
        self.y = None  # detrended Lake levels
        self.y1 = None  # detrended Bay levels
        self.wl = None  # lake levels
        self.wl1 = None  # bay levels
        self.wl2 = None  
        self.NumUniquePts = None  # Lake levels unique points
        self.NumUniquePts1 = None  # Bay levels unique points
        self.f = None  # Lake levels freq
        self.f1 = None  # Bay leevel frew
        self.f2 = None  # Bay leevel frew
        self.Time = None  # Lake levels time
        self.Time1 = None  # Bay levels time
        self.Time2 = None  # Bay levels time
        self.fftx = None  # fourier transform lake levels
        self.fftx1 = None  # fourier transform bay levels
        self.fftx2 = None  # fourier transform bay levels
        self.power = None  # power spectrum lake levels
        self.power1 = None  # power spectrum bay levels
        self.power2 = None  # power spectrum bay levels
        self.x05 = None  # 5% conf level lake
        self.x95 = None  # 95% conf level lake
        self.x05_1 = None  # 5% conf level bay
        self.x95_1 = None  # 95% conf level bay
        self.x05_2 = None  # 5% conf level bay
        self.x95_2 = None  # 95% conf level bay
        self.num_segments = 1
        self.data = data
        self.data1 = data1
        self.data2 = data2
        self.window = None
    # end

    def doSpectralAnalysis(self, showOrig, draw, tunits = 'sec', window = 'hanning', num_segments = 1, filter = None, \
                           log = 'linear', date1st = False, dateinterval=None, d3d=False):

        self.num_segments = num_segments
        self.window = window

        if self.filename != None:
            [self.wl, self.y, self.Time, self.fftx, self.NumUniquePts, self.mx, self.f, self.power, self.x05, self.x95] = \
            self.fftsa.FourierAnalysis(self.filename, showOrig, tunits = tunits, window = window, num_segments = num_segments, \
                                       filter = filter, log = log , bResample = False, Time1 = None, date1st = date1st, dateinterval=dateinterval, d3d=False)

            if self.filename1 != None:

                [self.wl1, self.y1, self.Time1, self.fftx1, self.NumUniquePts1, self.mx1, self.f1, self.power1, self.x05_1, self.x95_1] = \
                    self.fftsa.FourierAnalysis(self.filename1, showOrig, tunits, window, num_segments, filter, log, dateinterval=dateinterval )
                eps = (self.Time[1] - self.Time[0]) / 100

                if (self.Time[1] - self.Time[0]) - (self.Time1[1] - self.Time1[0]) > eps:
                    bResample = True
                    # [self.wl1, self.y1, self.Time1, self.fftx1, self.NumUniquePts1, self.mx1, self.f1, self.power1, self.x05_1, self.x95_1] = \
                    #    self.fftsa.FourierAnalysis(self.filename1, showOrig, tunits, window, num_segments, filter, log, bResample, self.Time)
                    [self.wl, self.y, self.Time, self.fftx, self.NumUniquePts, self.mx, self.f, self.power, self.x05, self.x95] = \
                        self.fftsa.FourierAnalysis(self.filename, showOrig, tunits, window, num_segments, filter, log, bResample, self.Time1)
                # end if
            # end if
        elif self.data is not None:
            [self.wl, self.y, self.Time, self.fftx, self.NumUniquePts, self.mx, self.f, self.power, self.x05, self.x95] = self.fftsa.FourierDataAnalysis(self.data, showOrig, draw, tunits, window, num_segments, log)
            if self.data1 is not None:
                [self.wl1, self.y1, self.Time1, self.fftx1, self.NumUniquePts1, self.mx1, self.f1, self.power1, self.x05_1, self.x95_1] = self.fftsa.FourierDataAnalysis(self.data1, showOrig, draw, tunits, window, num_segments, log)
            if self.data2 is not None:
                [self.wl2, self.y2, self.Time2, self.fftx2, self.NumUniquePts2, self.mx2, self.f2, self.power2, self.x05_2, self.x95_2] = self.fftsa.FourierDataAnalysis(self.data2, showOrig, draw, tunits, window, num_segments, log)   
                
        else:
            raise Exception("Both filename and data are missing ")

        return [self.Time, self.y, self.x05, self.x95, self.fftx, self.f, self.mx]
    # end doSpectralAnalysis


    def plotLakeLevels(self, lake_name, bay_name, detrend = False, x_label = None, y_label = None, title = None, plottitle = False, \
                       doy = False, grid = False, dateinterval=None):
        if self.show :
            # plot the original Lake oscillation input
            L = len(self.Time)
            if x_label == None:
                if dateinterval==None:
                    xlabel = 'Time [days]'
                else:
                    xlabel = 'Time [h]'
            else:
                xlabel = x_label
            if y_label == None:
                ylabel = 'Detrended Z [m]'
            else :
                ylabel = y_label

            if self.filename1 != None:
                xa = np.array([self.Time, self.Time])
                if detrend:
                    # detrending was already done
                    # self.y = fft_utils.detrend(self.y, 1)
                    # self.y1 = fft_utils.detrend(self.y1, 1)

                    ya = np.array([self.y, self.y1])
                else:
                    ya = np.array([self.wl, self.wl1])

                legend = [lake_name, bay_name]


            else:
                xa = np.array([self.Time])
                ya = np.array([self.y])
                legend = [lake_name]


            # end
            if title == None:
                title = "Detrended Lake Levels"
            else:
                title = title + " - time series"
            #if dateinterval != None:
            #    doy=False    
            fft_utils.plot_n_TimeSeries(title, xlabel, ylabel, xa, ya, legend = legend, plottitle = plottitle, doy = doy, grid = grid, dateinterval=dateinterval)

        # end if
    # end plotLakeLevels



    def plotSingleSideAplitudeSpectrumFreq(self, lake_name, bay_name, funits = "Hz", y_label = None, title = None,
                                            log = False, fontsize = 20, tunits = None, plottitle = False, grid = False, \
                                            ymax = None, graph = True):

        # smooth only if not segmented
        if self.num_segments == 1:
            sSeries = fft_utils.smoothSeries(self.mx, 5)
        else:
            sSeries = self.mx

        if self.filename1 != None and self.num_segments == 1:
            sSeries1 = fft_utils.smoothSeries(self.mx1, 5)
        else :
            sSeries1 = self.mx1
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            if title == None:
                title = 'Single-Sided Amplitude spectrum vs freq'
            else:
                title = title + " - Single-Sided Amplitude"

            if funits == 'Hz':
                xlabel = 'Frequency [Hz]'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency [cph]'
                f = self.f * 3600
            # end if

            if y_label == None:
                ylabel = '|Z(f)| [m]'
            else :
                ylabel = y_label

            if self.filename1 != None:
                xa = np.array([f, f])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end
            if graph:
                if self.num_segments == 1:
                    fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend = legend, log = log, plottitle = plottitle, ymax_lim = ymax)
                else:
                    fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel, xa, ya, ci05, ci95, legend = legend, \
                                                    log = log, fontsize = fontsize, plottitle = plottitle, grid = grid, ymax_lim = ymax)
            else:
                if self.num_segments == 1:
                    return [title, xlabel, ylabel, xa, ya, legend, log, plottitle, ymax_lim]
                else:
                    return [title, xlabel, ylabel, xa, ya, ci05, ci95, legend, log, fontsize, plottitle, ymax]
    # end plotSingleSideAplitudeSpectrumFreq


    def plotPSD2SpectrumsFreq(self, lake_name, bay_name, funits = "Hz", title = None,
                                         y_label1 = None, tunits = None,
                                         log = False, fontsize = 20, plottitle = False, grid = False, \
                                         ymax = None, graph = True,
                                         twoaxes = False, ylabel2 = None, ymax_lim2 = None, drawslope = False):

        sSeries1 = None
        # smooth only if not segmented
        if self.num_segments == 1:
            sSeries = fft_utils.smoothSeries(self.power, 5)
        else:
            sSeries = self.power

        if (self.filename1 != None or self.mx1 is not None):
            if self.num_segments == 1:
                sSeries1 = fft_utils.smoothSeries(self.power1, 5)
            else :
                sSeries1 = self.power1
        else:
            warnings.warn('Error: Second set of data not provided')
            return
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            if title == None:
                title = 'Single-Sided Amplitude spectrum vs freq'
            else:
                title = title + " - Single-Sided Amplitude"

            if funits == 'Hz':
                xlabel = 'Frequency [Hz]'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency [cph]'
                f = self.f * 3600
            # end if

            if y_label1 == None:
                ylabel1 = '|Z(f)| (m)'
            else :
                ylabel1 = y_label1

            if sSeries1 is not None:
                if funits == 'Hz':
                    f1 = self.f1
                elif funits == 'cph':
                    f1 = self.f1 * 3600
                # end if

                xa = np.array([f, f1])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end
            if graph:
                if self.num_segments == 1:
                    fft_utils.plot_n_Array(title, xlabel, ylabel1, xa, ya, legend = legend, log = log, plottitle = plottitle, ymax_lim = ymax,
                                           twoaxes = twoaxes, ylabel2 = ylabel2, ymax_lim2 = ymax_lim2)
                else:
                    ax = fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel1, xa, ya, ci05, ci95, legend = legend, \
                                                    log = log, fontsize = fontsize, plottitle = plottitle, grid = grid, ymax_lim = ymax,
                                                    twoaxes = twoaxes, ylabel2 = ylabel2, ymax_lim2 = ymax_lim2, drawslope = drawslope)
            else:
                if self.num_segments == 1:
                    return [title, xlabel, ylabel1, xa, ya, legend, log, plottitle, ymax_lim]
                else:
                    return [title, xlabel, ylabel1, xa, ya, ci05, ci95, legend, log, fontsize, plottitle, ymax]
    # end plotSingleSideAplitudeSpectrumFreq


    def plotPSD3SpectrumsFreq(self, lake_name, bay_name, third_name,  funits = "Hz", \
                                    log = False, fontsize = 20, plottitle = False, grid = True, \
                                    ymax = None, graph = True, tunits = None, ymax_lim2 = None, \
                                    ylabel1 = "Label1", ylabel2 = "Label2", ylabel3 = "Label3", \
                                    ymax_lim3 = None, drawslope = False, ispower = False):

        title=None
        sSeries1 = None
        sSeries2 = None
        # smooth only if not segmented
        if self.num_segments == 1:
            if ispower:
                sSeries = fft_utils.smoothSeries(self.power, 5)
            else:
                sSeries = fft_utils.smoothSeries(self.mx, 5)
        else:
            if ispower:
                sSeries = self.power
            else:
                sSeries = self.mx

        if (self.filename1 != None or self.mx1 is not None):
            if self.num_segments == 1:
                if ispower:
                    sSeries1 = fft_utils.smoothSeries(self.power1, 5)
                else:
                    sSeries1 = fft_utils.smoothSeries(self.mx1, 5)
            else :
                if ispower:
                    sSeries1 = self.power1
                else:
                    sSeries1 = self.mx1
        else:
            warnings.warn('Error: Second set of data not provided')
            return
        # end
        
        if (self.mx2 is not None):
            if self.num_segments == 1:
                if ispower:
                    sSeries2 = fft_utils.smoothSeries(self.power2, 5)
                else:
                    sSeries2 = fft_utils.smoothSeries(self.mx2, 5)
            else :
                if ispower:
                    sSeries2 = self.power2
                else:
                    sSeries2 = self.mx2
        else:
            warnings.warn('Error: Third set of data not provided')
            return
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.

            if funits == 'Hz':
                xlabel = 'Frequency [Hz]'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency [cph]'
                f = self.f * 3600
            # end if

            if ylabel1 == None:
                ylabel1 = '|Z(f)| [m]'
            else :
                ylabel1 = ylabel1

            if ylabel2 == None:
                ylabel2 = '|Z(f)| [m]'
            else :
                ylabel2 = ylabel2
            
            if ylabel3 == None:
                ylabel3 = '|Z(f)| [m]'
            else :
                ylabel3 = ylabel3

            if (sSeries1 is not None) and (sSeries2 is None):
                if funits == 'Hz':
                    f1 = self.f1
                elif funits == 'cph':
                    f1 = self.f1 * 3600
                # end if

                xa = np.array([f, f1])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]

            elif (sSeries1 is not None) and  (sSeries2 is not None):
                if funits == 'Hz':
                    f1 = self.f1
                    f2 = self.f2
                elif funits == 'cph':
                    f1 = self.f1 * 3600
                    f2 = self.f2 * 3600
                # end if
                xa = np.array([f, f1, f2])
                ya = np.array([sSeries, sSeries1, sSeries2])
                #xa = np.array([f, f1])
                #ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name, third_name]
                #legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1, self.x05_2]
                    ci95 = [self.x95, self.x95_1, self.x95_2]
                    #ci05 = [self.x05, self.x05_1]
                    #ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end
            if graph:
                if self.num_segments == 1:
                    fft_utils.plot_n_Array(title, xlabel, ylabel1, xa, ya, legend = legend, log = log, plottitle = plottitle, ymax_lim = ymax,
                                           twoaxes = twoaxes, ylabel2 = ylabel2, ymax_lim2 = ymax_lim2)
                else:
                    #===========================================================
                    # ax = fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel1, xa, ya, ci05, ci95, legend = legend, \
                    #                                 log = log, fontsize = fontsize, plottitle = plottitle, grid = grid, ymax_lim = ymax,\
                    #                                 twoaxes = False, ylabel2 = ylabel2, ymax_lim2 = ymax_lim2, drawslope = drawslope,\
                    #                                 threeaxes = True, ylabel3 = ylabel3, ymax_lim3 = ymax_lim3)
                    #===========================================================
                    ax = fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel1, xa, ya, ci05, ci95, legend = legend, \
                                                    log = log, fontsize = fontsize, plottitle = plottitle, grid = grid, ymax_lim = ymax,\
                                                    twoaxes = False, ylabel2 = ylabel2, ymax_lim2 = ymax_lim2, drawslope = drawslope,\
                                                    threeaxes = True, ylabel3 = ylabel3, ymax_lim3 = ymax_lim3)
            else:
                if self.num_segments == 1:
                    return [title, xlabel, ylabel1, xa, ya, legend, log, plottitle, ymax_lim]
                else:
                    return [title, xlabel, ylabel1, xa, ya, ci05, ci95, legend, log, fontsize, plottitle, ymax]
    # end plotSingleSideAplitudeSpectrumFreq



    def compute_CI(self, data, ci = 0.95, log = False):
        # for power density the confidence interval calculationw are not good as they are for amplitude. NEED to RECALCULATE
        interval_len = len(self.Time) / self.num_segments
        data_len = len(self.Time)
        edof = fft_utils.edof(data, data_len, interval_len, self.window)  # one dt chunk see Hartman Notes ATM 552 page 159 example
        (x05, x95) = fft_utils.confidence_interval(data, edof, 0.95, log)
        return (x05, x95)

    def plotPowerDensitySpectrumFreq(self, lake_name, bay_name, funits = "Hz", y_label = None, title = None, \
                                     log = False, fontsize = 20, tunits = None, plottitle = False, grid = False):

        # smooth only if not segmented
        if self.num_segments == 1:
            sSeries = fft_utils.smoothSeries(np.abs(self.power), 5)
        else:
            sSeries = self.power
            # for power density the confidence interval calculationw are not good as they are for amplitude. NEED to RECALCULATE
            (self.x05, self.x95) = self.compute_CI(np.abs(self.power), 0.95, log)


        if self.filename1 != None and self.num_segments == 1:
            sSeries1 = fft_utils.smoothSeries(self.power1, 5)
        elif self.filename1 != None:
            sSeries1 = self.power1
            # for power density the confidence interval calculationw are not good as they are for amplitude. NEED to RECALCULATE
            (self.x05_1, self.x95_1) = self.compute_CI(np.abs(self.power1), 0.95, log)
        # end



        if self.show:
            # Plot single - sided amplitude spectrum.
            if title == None:
                title = 'Power Density spectrum vs freq'
            else:
                title = title + " - Power Density Spectrum"

            if funits == 'Hz':
                xlabel = 'Frequency [Hz]'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency [cph]'
                f = self.f * 3600
            # end if

            if y_label == None:
                ylabel = 'PSD (m$^2$/' + funits + ')'
            else :
                ylabel = y_label

            if self.filename1 != None:
                xa = np.array([f, f])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end
            
            if self.num_segments == 1:
                fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, log, plottitle, grid = grid)
            else:
                fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel, xa, ya, ci05, ci95, legend = legend, \
                                               log = log, fontsize = fontsize, plottitle = plottitle, grid = grid)

    # end plotPowerDensitySpectrumFreq

    def plotPSDFreq(self, lake_name, bay_name, funits = "Hz", y_label = None, title = None, log = False, fontsize = 20, \
                    tunits = None, plottitle = False):

        # smooth only if not segmented
        if self.num_segments == 1:
            sSeries = fft_utils.smoothSeries(self.power, 5)
        else:
            sSeries = self.power
            # for power density the confidence interval calculationw are not good as they are for amplitude. NEED to RECALCULATE
            (self.x05, self.x95) = self.compute_CI(np.abs(self.power), 0.95, log)


        if self.filename1 != None and self.num_segments == 1:
            sSeries1 = fft_utils.smoothSeries(self.power1, 5)
        else :
            sSeries1 = self.power1
            # for power density the confidence interval calculationw are not good as they are for amplitude. NEED to RECALCULATE
            (self.x05_1, self.x95_1) = self.compute_CI(np.abs(self.power1), 0.95, log)
        # end


        if self.show:
            # Plot single - sided amplitude spectrum.
            if title == None:
                title = 'Power Density spectrum vs freq'
            else:
                title = title + " - Power Density Spectrum"

            if funits == 'Hz':
                xlabel = 'Frequency [Hz]'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency [cph]'
                f = self.f * 3600
            # end if

            if y_label == None:
                ylabel = 'PSD (m$^2$/' + funits + ')'
            else :
                ylabel = y_label

            if self.filename1 != None:
                xa = np.array([f, f])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end


        NFFT = len(sSeries)
        wlen = int(NFFT / self.num_segments)
        noverlap = int (wlen / 2)  # 50% overlap
        # prepare for the amplitude spectrum analysis
        if tunits == 'day':
            factor = 86400
        elif tunits == 'hour':
            factor = 3600
        else:
            factor = 1
        dt_s = (self.Time[2] - self.Time[1]) * factor  # Sampling period [s]
        Fs = 1 / dt_s  # Samplig freq    [Hz]
        plt.psd(sSeries, NFFT = NFFT, Fs = Fs, window = np.hanning(NFFT), sides = 'onesided', noverlap = noverlap, pad_to = None, scale_by_freq = True)
        if plottitle:
            plt.title('Welch')
        plt.ylabel(y_label)
        plt.grid(True)

        plt.show()

    def plotSingleSideAplitudeSpectrumTime(self, lake_name, bay_name, y_label = None, title = None, \
                                           ymax_lim = None, log = False, tunits = None, plottitle = False, grid = False):
        sSeries = fft_utils.smoothSeries(self.mx, 5)
        if self.filename1 != None:
            sSeries1 = fft_utils.smoothSeries(self.mx1, 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            if title == None:
                title = 'Single-Sided Amplitude spectrum vs time [h]'
            else:
                title = title + " - Single-Sided Amplitude"
            xlabel = 'Time (h)'
            if y_label == None:
                ylabel = '|Z(f)| (m)'
            else :
                ylabel = y_label
            if self.filename1 != None:
                tph = (1 / self.f[1:]) / 3600
                tph1 = (1 / self.f1[1:]) / 3600
                xa = np.array([tph, tph])
                ya = np.array([sSeries[1:], sSeries1[1:]])
                legend = [lake_name, bay_name]
            else:
                tph = (1 / self.f[1:]) / 3600
                xa = np.array([tph])
                ya = np.array([sSeries[1:]])
                legend = [lake_name]
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, linewidth = 1.2, \
                                   ymax_lim = ymax_lim, log = log, plottitle = plottitle)
    # end plotSingleSideAplitudeSpectrumTime

    def plotZoomedSingleSideAplitudeSpectrumFreq(self, plottitle = False):
        sSeries = fft_utils.smoothSeries(self.mx[100:-1], 5)
        if self.filename1 != None:
            sSeries1 = fft_utils.smoothSeries(self.mx1[100:-1], 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Zoomed Single-Sided Amplitude spectrum vs freq'
            xlabel = 'Frequency [Hz]'
            ylabel = '|Z(f)| [m]'
            if self.filename1 != None:
                xa = np.array([self.f[100:-1], self.f[100:-1]])
                ya = np.array([sSeries, sSeries1])
                legend = ['lake', 'bay']
            else:
                xa = np.array([self.f[100:-1]])
                ya = np.array([sSeries])
                legend = ['lake']
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, plottitle = plottitle)
    # end plotZoomedSingleSideAplitudeSpectrumFreq

    def plotZoomedSingleSideAplitudeSpectrumTime(self, plottitle = False):
        zsSeries = fft_utils.smoothSeries(self.mx[100:-1], 5)
        if self.filename1 != None:
            zsSeries1 = fft_utils.smoothSeries(self.mx1[100:-1], 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Zoomed Single-Sided Amplitude spectrum vs time [h]'
            xlabel = 'Time [h]'
            ylabel = '|Z(f)| [m]'
            if self.filename1 != None:
                tph = (1 / self.f[100:-1]) / 3600
                tph1 = (1 / self.f1[100:-1]) / 3600
                xa = np.array([tph, tph])
                ya = np.array([zsSeries, zsSeries1])
                legend = ['lake', 'bay']
            else:
                tph = (1 / self.f[100:-1]) / 3600
                xa = np.array([tph])
                ya = np.array([zsSeries])
                legend = ['lake']
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, plottitle = plottitle)
    # end plotZoomedSingleSideAplitudeSpectrumTime

    def plotCospectralDensity(self, title = None, x_label = None, y_label = None, \
                              funits = "cph", plottitle = False, log = False):
        # plot the power of the cospectral density F_in(w) * F_out(w)
        #
        zsSeries = fft_utils.smoothSeries(self.mx, 5)
        if self.filename1 != None:
            zsSeries1 = fft_utils.smoothSeries(self.mx1, 5);
            convolution = zsSeries * zsSeries1.conjugate()
            ya = np.array([ convolution])
            legend = ['Cospectrum']

            if self.show:
                # Plot single - sided amplitude spectrum.
                if title == None:
                    title = 'Cospectral Power Density spectrum vs freq'
                else:
                    title = title + " - Cospectral Power Density Spectrum"

                if funits == 'Hz':
                    xlabel = 'Frequency [Hz]'
                    f = self.f
                elif funits == 'cph':
                    xlabel = 'Frequency [cph]'
                    f = self.f * 3600
                # end if
                xa = np.array([f])

                if y_label == None:
                    ylabel = 'PSD [m$^2$/' + funits + ']'
                else :
                    ylabel = y_label

                fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, log, plottitle)

            # end
        # end
    # end plotCospectralDensity

    def plotPhase(self, plottitle = False):
        if self.show:
            # phase = np.unwrap(np.angle(self.fftx[0:self.NumUniquePts]))
            phase = np.unwrap(np.angle(self.fftx[0:len(self.fftx) / 2 + 1]))
            tph = (1.0 / self.f) / 3600
            xlabel = 'Time period [h]'
            ylabel = 'Phase [Degrees]'
            title = 'Phase delay'
            legend = [ 'phase']
            # avoind plotting the inf value of tph and start from index 1.
            fft_utils.plot_n_Array(title, xlabel, ylabel, [tph[1:]], [phase[1:] * 180 / np.pi], legend, plottitle = plottitle)
        # end
    # end plotPhase

