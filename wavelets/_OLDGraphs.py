'''
Created on Jun 19, 2012

@author: bogdan
'''


import numpy as np
import math
import mlpy.wavelet as wave
from . import cwt

import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import seconds
from matplotlib.dates import date2num, num2date
from matplotlib.dates import MONDAY, SATURDAY
import matplotlib.dates

years = matplotlib.dates.YearLocator()  # every year
months = matplotlib.dates.MonthLocator()  # every month
yearsFmt = matplotlib.dates.DateFormatter('%Y')  # every monday
mondays = matplotlib.dates.WeekdayLocator(MONDAY)

class Graphs(object):
    '''
    classdocs
    '''


    def __init__(self, path, file1, file2, show):
        '''
        Constructor
        '''
        self.path_in = path
        self.filename1 = file1
        self.filename2 = file2
        self.show = show
        self.cwtObj = cwt.Cwt(path, file1, file2)
        self.Time = None
        self.lakeWavelet = None
        self.bayWavelet = None
        self.scalesLake = None
        self.scalesBay = None
        self.freqLake = None
        self.corrLake = None
        self.freqBay = None
        self.corrBay = None
        self.SensorDepthLake = None
        self.SensorDepthBay = None

    def doSpectralAnalysis(self):
        a, b = self.cwtObj.doSpectralAnalysis()
        [self.Time, self.SensorDepthLake, self.lakeWavelet, self.scalesLake, self.freqLake] = a
        if  self.filename2 != None:
           [self.Time, self.SensorDepthBay, self.bayWavelet, self.scalesBay, self.freqBay] = b
        return [a, b]

    def showGraph(self):
        plt.show()

    def plotDateScalogram(self, scaleType = None, plotFreq = True, printtitle = False):

        fig, ax1, ax2, ax3 = self.plotScalogram(scaleType, plotFreq)
        formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(formatter)
        ax1.xaxis.set_minor_locator(mondays)
        ax1.xaxis.grid(True, 'minor')

        ax2.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_minor_locator(mondays)
        ax2.xaxis.grid(True, 'minor')
        fig.autofmt_xdate()

    def plotScalogram(self, scaleType = None, plotFreq = True, printtitle = False):
        # approximate scales through frequencies
        # freq = (omega0 + np.sqrt(2.0 + omega0 ** 2)) / (4 * np.pi * scale[1:])

        fig = plt.figure()  # creating an empty Matplotlib figure

        title = "Wavelets - Scalogram"
        # axis for the initial signal
        ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])

        # axis for the resulting wavelet transform data
        if plotFreq == True:
            ylabel = "Period (hours)"
        else:
            ylabel = "Scales"

        ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], xlabel = "Time", ylabel = ylabel, title = title)

        if scaleType != None:
            if scaleType == 'log':
                ax2.set_yscale(scaleType)

        ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6], xlabel = "Magnitude")  # axis for a colour bar
        ax1.plot(self.Time, self.SensorDepthLake, 'k')  # plotting the initial time series

        # plotting a CWT data
        if plotFreq == True:
            img = ax2.pcolormesh(self.Time, self.freqLake, np.abs(self.lakeWavelet))
        else:
            img = ax2.imshow(np.abs(self.lakeWavelet), extent = [self.Time[0], self.Time[-1], self.scalesLake[-1], self.scalesLake[0]], aspect = 'auto')
        fig.colorbar(img, cax = ax3)  # building a colour bar basing on the colours present in CWT data image
        # format the ticks
        ax2.grid(True)
        ax1.axis('tight')
        ax2.axis('tight')
        if printtitle:
            ax1.set_title(title)
        plt.draw()
        return [fig, ax1, ax2, ax3]

    def plotSingleSideAplitudeSpectrumTime(self, printtitle = False):
        fig = plt.figure()
        power1 = (np.abs(self.lakeWavelet)) ** 2

        plt.title('Lake Amplitude (m)')
        A1 = np.sqrt(np.sum(power1, axis = 1) / self.SensorDepthLake.shape[0])
        plt.plot(self.freqLake, A1)  # * self.corrLake)
        if self.filename2 != None:
            if printtitle:
                plt.title('Wavelets - Lake and Bay Amplitude (m) / Time')
            power2 = (np.abs(self.bayWavelet)) ** 2
            A2 = np.sqrt(np.sum(power2, axis = 1) / self.SensorDepthBay.shape[0])
            plt.ylabel("Amplitude (m)")
            plt.xlabel("Time (hours)")
            plt.plot(self.freqBay, A2)  # * self.corrBay)
            plt.legend(['Lake', 'Bay'])
        else:
            plt.legend(["Lake"])

    def plotSingleSideAplitudeSpectrumFreq(self, printtitle = False):
        fig = plt.figure()
        power1 = (np.abs(self.lakeWavelet)) ** 2

        plt.title('Wavelets - Lake Amplitude (m)')
        A1 = np.sqrt(np.sum(power1, axis = 1) / self.SensorDepthLake.shape[0])
        plt.plot(1 / self.freqLake / 3600, A1)  # * self.corrLake)

        if self.filename2 != None:
            plt.title('Wavelets - Lake and Bay Amplitude (m) /Freq')
            power2 = (np.abs(self.bayWavelet)) ** 2
            A2 = np.sqrt(np.sum(power2, axis = 1) / self.SensorDepthBay.shape[0])
            plt.ylabel("Amplitude (m)")
            plt.xlabel("Freq (Hz)")
            plt.plot(1 / self.freqBay / 3600, A2)  # * self.corrBay)
            plt.legend(['Lake', 'Bay'])
        else:
            plt.legend(['Lake'])
    # end plotSingleSideAplitudeSpectrumTime



if __name__ == '__main__':
    '''
    Testing ground for local functions

    '''
    path = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/LakeOntario-data'
    graph = Graphs(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv', True)
    graph.doSpectralAnalysis()
    graph.plotDateScalogram(plotFreq = True)
    graph.plotSingleSideAplitudeSpectrumTime()
    graph.plotSingleSideAplitudeSpectrumFreq()
    graph.showGraph()


