'''
Created on Jun 12, 2012

@author: bogdan
'''
import matplotlib.pyplot as plt
from datetime import datetime
#from datetime import timedelta
#from matplotlib.dates import seconds
from matplotlib.dates import num2date
from matplotlib.dates import MONDAY #, SATURDAY
#import matplotlib.mlab
#import matplotlib.dates as dates
#import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import numpy as np
import scipy as sp
import scipy.signal
from scipy import  stats
import math
from ufft import smooth
#import time, os
import csv
#import chi2
from ufft import filters
import matplotlib.ticker

years = matplotlib.dates.YearLocator()  # every year
months = matplotlib.dates.MonthLocator()  # every month
yearsFmt = matplotlib.dates.DateFormatter('%Y')
mondays = matplotlib.dates.WeekdayLocator(MONDAY) # every monday



class EmbeddedSciFormatter(matplotlib.ticker.Formatter):
    """Provides a format for embedded-scientific notation."""

    def __init__(self, order_of_magnitude, digits):
        """ORDER_OF_MAGNITUDE is the power of ten to drag out.  DIGITS is the
        number of digits after to point to maintain."""

        self.order_of_magnitude = order_of_magnitude

        # Calculate the format, e.g. '$%f.2\ 10^{5}' .
        #self.format = '$%.' + '%d' % digits + r'f\times 10^{%d}$' % \
        self.format = '$%.' + '%d' % digits + r'f\cdot 10^{%d}$' % \
                order_of_magnitude

    def __call__(self, value, pos_dummy):
        """Returns the formatted value, in math text."""

        # Drag out the order of magnitude.
        reduced_value = value * 10 ** (-self.order_of_magnitude)
        return self.format % reduced_value


def timestamp2doy(dateTime):
    '''
    Converts from date time in seconds to day of the year
    '''
    dofy = np.zeros(len(dateTime))
    for j in range(0, len(dateTime)) :
        d = num2date(dateTime[j])
        dofy[j] = d.timetuple().tm_yday + d.timetuple().tm_hour / 24. + d.timetuple().tm_min / (24. * 60) + d.timetuple().tm_sec / (24. * 3600)

    return dofy

def select_interval_data(dateTime, data, dateinterval, d3d=True):
    '''
    Selects data corresponding to the dates 
    '''
    print ("DATE1 %s DATE2: %s" %(dateinterval[0],dateinterval[1]))
           
    dt1 = datetime.strptime(dateinterval[0], "%y/%m/%d %H:%M:%S")
    start_num = matplotlib.dates.date2num(dt1)
    dt2 = datetime.strptime(dateinterval[1], "%y/%m/%d %H:%M:%S")
    end_num = matplotlib.dates.date2num(dt2)
    
    sel_data = []
    sel_time = []
    firstdate=0
    for t,d in zip(dateTime,data):
        if t >= start_num and t <= end_num:
        #if t >= start_num and t < end_num:
            if d3d==0:
                firstdate=t
                #print ("appended:%f data:%f" % (t, d))
                sel_time.append((t-firstdate)*24)
            else:
                sel_time.append(t)
            sel_data.append(d)

    print(sel_time)
    print(sel_data)
    return np.array(sel_time), np.array(sel_data)



def drange(start, stop, step):
    '''
    Example:
    i0=drange(0.0, 1.0, 0.1)
    >>>["%g" % x for x in i0]
    ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    '''

    r = start
    while r < stop:
        yield r
        r += step
    # end while
# end drange


def readFile(path_in, fname, date1st = False, sep=","):
    # read Lake data
    filename = path_in + '/' + fname

    ifile = open(filename, 'rt')
    reader = csv.reader(ifile, delimiter = sep, quotechar = '"')
    rownum = 0
    SensorDepth = []
    Time = []
    if  date1st:
        if sep == " ":
            ix1 = 1
            ix2 = 0
        else:
            ix1 = 2
            ix2 = 1
    else:
        ix1 = 1
        ix2 = 0

    printHeaderVal = False
    for row in reader:
        try:
            #if filename == ' /home/bogdan/Documents/UofT/PhD/Data_Files/2013/Hobo-Apr-Nov-2013/WL/csv_press_corr/01-13-Aug-WL-Spadina_out.csv':
            #    print("EXCEPTION  artificial division of data by 5 due to model boundaries input were given from Inn Harbour which are already amplified")
            #    SensorDepth.append(float(row[ix1])/5)
            #else:
            SensorDepth.append(float(row[ix1]))
            Time.append(float(row[ix2]))
        except:
            pass
    # end for

    return [Time, SensorDepth]
# end readFile

def moving_average(x, n, type = 'simple'):
    """
    compute an n period moving average.
    type is 'simple' | 'exponential'
    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()
    a = np.convolve(x, weights, mode = 'full')[:len(x)]
    a[:n] = a[n]
    return a


def nextpow2(i):
    """
    Find the next power of two

    >>> nextpow2(5)
    8
    >>> nextpow2(250)
    256
    """
    # do not use numpy here, math is much faster for single values
    buf = math.ceil(math.log(i) / math.log(2))
    return int(math.pow(2, buf))
# end nextpow2

def smoothSeries(data, span):
    '''
    '''
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    results_temp = smooth.smoothfit(data, span, windows[2])

    return results_temp['smoothed']
# end smoothSeries


# another moving average smoothing similar to self.moving _average()
def smoothSeriesWindow(data, WINDOW = 10):

    extended_data = np.hstack([[data[0]] * (WINDOW - 1), data])
    weightings = np.repeat(1.0, WINDOW) / WINDOW
    smoothed = np.convolve(extended_data, weightings)[WINDOW - 1:-(WINDOW - 1)]
    return smoothed

def smoothListGaussian(list, strippedXs = False, degree = 100):



    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weightGauss = []

    for i in range(window):

        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):

        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return smoothed

# This is a moving average detrend
def detrend(data, degree = 10):
        detrended = [None] * degree
        for i in range(degree, len(data) - degree):
                chunk = data[i - degree:i + degree]
                chunk = sum(chunk) / len(chunk)
                detrended.append(data[i] - chunk)
        return detrended + [None] * degree
# end detrend

# This is a similar to sp.signal.detrend
def detrend_separate(y, order = 0):
    '''detrend multivariate series by series specific trends

    Paramters
    ---------
    y : ndarray
       data, can be 1d or nd. if ndim is greater then 1, then observations
       are along zero axis
    order : int
       degree of polynomial trend, 1 is linear, 0 is constant

    Returns
    -------
    y_detrended : ndarray
       detrended data in same shape as original

    '''
    nobs = y.shape[0]
    shape = y.shape
    y_ = y.reshape(nobs, -1)
    kvars_ = len(y_)
    t = np.arange(nobs)
    exog = np.vander(t, order + 1)
    params = np.linalg.lstsq(exog, y_)[0]
    fittedvalues = np.dot(exog, params)
    resid = (y_ - fittedvalues).reshape(*shape)
    return resid, params

def plotArray(title, xlabel, ylabel, x, y, legend = None, linewidth = 0.6, plottitle = False):

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        ax = fig.add_subplot(111)

        ax.plot(x, y, linewidth = 0.6)

        ax.xaxis.grid(True, 'minor')
        ax.grid(True)
        title = title

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if plottitle:
            plt.title(title)
        if legend != None:
            plt.legend(legend);

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        plt.draw()
        plt.show()
    # end


def autoscale_based_on(ax, vertices):
    # ax.dataLim = mtransforms.Bbox.unit()
    # for line in lines:
    #    xy = np.vstack(line.get_data()).T
    #    ax.dataLim.update_from_data_xy(xy, ignore = False)
    vertices = np.array(vertices)
    ax.dataLim.update_from_data_xy(vertices, ignore = False)
    ax.autoscale_view()


def plot_n_Array(title, xlabel, ylabel, x_arr, y_arr, legend = None, linewidth = 0.6, ymax_lim = None, log = 'linear', \
                 plottitle = False, grid = False, fontsize = 18, noshow = False, twoaxes = False, ylabel2 = None, ymax_lim2 = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    lines = []
    ax = fig.add_subplot(111)

    if twoaxes:
        ax2 = ax.twinx()

    i = 0;
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    minx = 1e10
    maxx = 0
    miny = 1e10
    maxy = 0
    for a in x_arr:
        x = x_arr[i]
        y = y_arr[i]
        maxx = np.max(x)
        minx = np.min(x)
        maxy = np.max(y)
        miny = np.min(y)

        if twoaxes and i == 1:
            axs = ax2
        else:
            axs = ax
        if log=='loglog':
            line = axs.loglog(x, y, linestyle = ls[i], linewidth = 1.2 + 0.4 * i, basex = 10)
        elif log=='log':
            line = axs.semilogx(x, y, linestyle = ls[i], linewidth = 1.2 + 0.4 * i, basex = 10)
        else:
            line = axs.plot(x, y, linestyle = ls[i], linewidth = 1.2 + 0.4 * i)

        lines.append(line)
        i += 1
    # end for
    vertices = [(minx, miny), (maxx, maxy)]
    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(grid)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)

    if twoaxes and ylabel2 != None :
        ax2.set_ylabel(ylabel2, fontsize = fontsize)

    if plottitle:
        plt.title(title).set_fontsize(fontsize + 2)

    if legend is not None:
        plt.legend(legend);
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room fornumpy smoothing filter them

    if ymax_lim != None:
        plt.ylim(ymax = ymax_lim)
    else:
        autoscale_based_on(ax, vertices)
    if not noshow:
        plt.show()
    return ax
# end

def errorbar(ax, x0, y0, ci, color):
    #ax.loglog([x0, x0], [y0 * ci[0], y0 * ci[1]], color = color)
    ax.loglog([x0, x0], [y0 + ci[0], y0 + ci[1]], color = color)
    ax.loglog(x0, y0, 'bo')
    #ax.loglog(x0, y0 * ci[0], 'b_')
    #ax.loglog(x0, y0 * ci[1], 'b_')
    ax.loglog(x0, y0 + ci[0], 'b_')
    ax.loglog(x0, y0 + ci[1], 'b_')


def errorbarsemilogx(ax, x0, y0, ci, color):
    ax.semilogy([x0, x0], [y0 + ci[0], y0 + ci[1]], color = color)
    ax.semilogyloglog(x0, y0, 'bo')
    ax.semilogy(x0, y0 + ci[0], 'b_')
    ax.semilogy(x0, y0 + ci[1], 'b_')

def set_axis_type(ax, type):
    if type == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
        print("xscale=log; yscale=log")
        # ax[i].tick_params(axis='x', labelsize=18)

    elif type == 'log':
        ax.set_xscale('log')
        ax.set_yscale('linear')
        print("xscale=linear; yscale=log")
    else:
        print("xscale=linear; yscale=linear")


def plot_n_Array_with_CI_twinx(title, xlabel, ylabel, x_arr, y_arr, ci05, ci95, legend = None, linewidth = 0.8, ymax_lim = None, log = 'linear', \
                         fontsize = 20, plottitle = False, grid = False, twoaxes = False, ylabel2 = None, ymax_lim2 = None, drawslope = False,\
                          threeaxes=False, ylabel3 = None, ymax_lim3 = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    print ("==> In plot_n_Array_with_CI <== fontsize:%d" % fontsize)
    ax=[]
    sz=1
    #ax1 = host_subplot(111, axes_class=AA.Axes)
    ax1 = fig.add_subplot(111)
    # plt.subplots_adjust(right=0.85)
    set_axis_type(ax1, log)
    ax1.set_ylabel(ylabel, fontsize=fontsize-2)
    ax1.set_xlabel(xlabel, fontsize=fontsize-2)
    ax1.tick_params(labelsize=fontsize - 5)
    ax.append(ax1)

    if  twoaxes or threeaxes:
        sz = 2
        ax2 = ax1.twinx()
        ax.append(ax2)
        set_axis_type(ax2, log)
        ax2.set_ylabel(ylabel2, fontsize=fontsize-2)
        ax2.set_xlabel(xlabel, fontsize=fontsize-2)
        ax2.tick_params(labelsize=fontsize - 5)
    if threeaxes:
        sz = 3
        ax3 = ax1.twinx()
        ax.append(ax3)
        set_axis_type(ax3, log)
        ax3.set_ylabel(ylabel3, fontsize=fontsize-2)
        ax3.set_xlabel(xlabel, fontsize=fontsize-2)
        ax3.tick_params(labelsize=fontsize - 5)
        offset = 70
        new_fixed_axis = ax[2].get_grid_helper().new_fixed_axis
        ax[2].axis["right"] = new_fixed_axis(loc="right", axes=ax[2],offset=(offset, 0))
        ax[2].axis["right"].toggle(all=True)   
    #endif

    lines = []
    Xmax = []
    Ymax = []
    Xmin = []
    Ymin = []
 
    lst = ['-', '--', '-.', ':', '-', '--', ':', '-.']
    lst = ['-', '-', '-', '-', '-', '-', '-', '-']
    colors = ['b', 'c', 'r', 'k', 'y', 'm', 'aqua', 'k']
    
    # plot the confidence intervals
    if len(ci05) > 0:
        i = 0
        xc = 0
        for a in ci05:  # x_arr:
            arry = hasattr(a, "__len__")
            x = x_arr[i][3:]
            y = y_arr[i][3:]
            
            Xmax.append(np.max(x))
            Xmin.append(np.min(x))

            if (twoaxes and i == 1) or (threeaxes and i == 1):
                axs = ax[1]
            elif threeaxes and i == 2:
                axs = ax[2]
            else:
                axs = ax[0]

            y1 = ci05[i][3:]
            y2 = ci95[i][3:]
            x = x_arr[i][3:]
            y = y_arr[i][3:]

            if len(x_arr) < 5:
                lwt = 1.6 - i * 0.2
            else:
                lwt = 1 + i * 0.6

            sd = 0.65 - i * 0.15
            ymax = max(np.max(y1), np.max(y2))
            ymin = min(np.min(y1), np.min(y2))

            axs.fill_between(x, y1, y2, where = y2 > y1, facecolor = [sd, sd, sd], alpha = 0.5, interpolate = True, linewidth = 0.001)
            line = axs.plot(x, y, linestyle=lst[i], linewidth=lwt, color=colors[i], label=legend[i])
            lines += line
            if len(Ymin) <  i+1:
                Ymin.append(ymin)
            else:
                Ymin[i] = min(Ymin[i], ymin)
            if len(Ymax) <  i+1:
                Ymax.append(ymax)
            else:
                Ymax[i] = max(Ymax[i], ymax)
            
            i += 1
        #end for 
    # end if len(ci)

    #set limits
    for i in range(0, len(ax)):
        vertices = [(Xmin[i], Ymin[i]), (Xmax[i], Ymax[i])]
        ax[i].xaxis.grid(grid, 'minor')
        ax[i].yaxis.grid(grid, 'minor')
        ax[i].grid(grid)
        if i  == 0 :
            side = 'left'
        else:
            side = "right"
        """
        if  twoaxes or threeaxes:
            ax[i].axis[side].label.set_fontsize(fontsize - 4)
            ax[i].axis["bottom"].label.set_fontsize(fontsize - 4)
            ax[i].axis[side].major_ticklabels.set_fontsize(fontsize - 7)
            ax[i].axis["bottom"].major_ticklabels.set_fontsize(fontsize - 7)
        
        for tick in ax[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize-4)
        for tick in ax[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize-4)
        """
    plt.xlim(xmin=np.min(Xmin), xmax=np.max(Xmax))

    if plottitle:
            plt.title(title, fontsize = fontsize)

    if legend is not None:
        labs = [l.get_label() for l in lines]
        # loc=4      position lower right
        #---------------------------------------------------------- upper right    1
        #---------------------------------------------------------- upper left    2
        #---------------------------------------------------------- lower left    3
        #---------------------------------------------------------- lower right    4
        #---------------------------------------------------------- right    5
        #------------------------------------------------------ --- center left    6
        #---------------------------------------------------------- center right    7
        #---------------------------------------------------------- lower center    8
        #---------------------------------------------------------- upper center    9
        legnd = ax[0].legend(lines, labs, loc=3, frameon=False)
        for label in legnd.get_texts():
            label.set_fontsize(fontsize-6)

    plt.show()
# end


#------------------------------------------------------------------------------------------------------------------------
def plot_n_Array_with_CI(title, xlabel, ylabel, x_arr, y_arr, ci05, ci95, legend=None, linewidth=0.8, ymax_lim=None,
                         log='linear', \
                         fontsize=20, plottitle=False, grid=False, twoaxes=False, ylabel2=None, ymax_lim2=None,
                         drawslope=False, \
                         threeaxes=False, ylabel3=None, ymax_lim3=None):
    fig = plt.figure(facecolor='w', edgecolor='k')

    print("==> In plot_n_Array_with_CI <== fontsize:%d" % fontsize)
    ax = []
    Y_labels = []
    Y_labels.append(ylabel)
    sz = 1
    if twoaxes or threeaxes:
        sz = 2
        ax.append(host_subplot(111, axes_class=AA.Axes))
        plt.subplots_adjust(right=0.75)
    else:
        ax.append(fig.add_subplot(111))
    if twoaxes or threeaxes:
        ax.append(ax[0].twinx())
        Y_labels.append(ylabel2)
    if threeaxes:
        sz = 3
        ax.append(ax[0].twinx())
        offset = 70
        new_fixed_axis = ax[2].get_grid_helper().new_fixed_axis
        ax[2].axis["right"] = new_fixed_axis(loc="right", axes=ax[2], offset=(offset, 0))
        ax[2].axis["right"].toggle(all=True)
        Y_labels.append(ylabel3)
    # endif
    for i in range(0, sz):
        if log == 'loglog':
            # line = axs.loglog(x, y, linestyle = lst[i], linewidth = lwt, basex = 10, color = colors[i],
            # label = legend[i])
            # axs.set_yscale('log')
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            # axs.semilogx(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i])
            print("xscale=log; yscale=log")
        elif log == 'log':
            ax[i].set_xscale('log')
            ax[i].set_yscale('linear')
            # line = axs.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i], label = legend[i])
            print("xscale=linear; yscale=log")
        else:
            # formatter =  EmbeddedSciFormatter(10, 1)
            # axs.yaxis.set_major_formatter(formatter)
            # line = axs.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i], label = legend[i])
            print("xscale=linear; yscale=linear")

    lines = []
    Xmax = []
    Ymax = []
    Xmin = []
    Ymin = []

    lst = ['-', '--', '-.', ':', '-', '--', ':', '-.']
    lst = ['-', '-', '-', '-', '-', '-', '-', '-']
    #colors = ['b', 'c', 'r', 'k', 'y', 'm', 'aqua', 'k']
    colors = ['r', 'c', 'b', 'k', 'y', 'm', 'aqua', 'k']
    # plot the confidence intervals
    if len(ci05) > 0:
        i = 0
        xc = 0
        for a in ci05:  # x_arr:
            arry = hasattr(a, "__len__")
            x = x_arr[i][3:]
            y = y_arr[i][3:]

            Xmax.append(np.max(x))
            Xmin.append(np.min(x))

            if (twoaxes and i == 1) or (threeaxes and i == 1):
                axs = ax[1]
            elif threeaxes and i == 2:
                axs = ax[2]
            else:
                axs = ax[0]

            y1 = ci05[i][3:]
            y2 = ci95[i][3:]
            sd = 0.65 - i * 0.15
            ymax = max(np.max(y1), np.max(y2))
            ymin = min(np.min(y1), np.min(y2))

            # line = axs.plot(x, y1, x, y2, color = [sd, sd, sd], alpha = 0.5)
            # print y2 > y1
            axs.fill_between(x, y1, y2, where=y2 > y1, facecolor=[sd, sd, sd], alpha=0.5, interpolate=True,
                             linewidth=0.001)
            # axs.fill_between(x, y1, y2, facecolor = 'blue', alpha = 0.5, interpolate = True)
            if len(Ymin) < i + 1:
                Ymin.append(ymin)
            else:
                Ymin[i] = min(Ymin[i], ymin)
            if len(Ymax) < i + 1:
                Ymax.append(ymax)
            else:
                Ymax[i] = max(Ymax[i], ymax)

            i += 1
        # end for
    # end if len(ci)

    i = 0
    for a in x_arr:
        x = x_arr[i][3:]
        y = y_arr[i][3:]

        if len(x_arr) < 5:
            lwt = 1.6 - i * 0.2
        else:
            lwt = 1 + i * 0.6

        if threeaxes and i == 2:
            axs = ax[2]
        if (threeaxes and i == 1) or (twoaxes and i == 1):
            axs = ax[1]
        else:
            axs = ax[0]

        line = axs.plot(x, y, linestyle=lst[i], linewidth=lwt, color=colors[i], label=legend[i])

        if drawslope:
            # select only the data we need to asses
            x1 = 0.1;
            x2 = 0.8
            xsl = []
            ysl = []
            for j in range(0, len(x)):
                if x[j] >= x1 and x[j] <= x2:
                    xsl.append(x[j])
                    ysl.append(y[j])
            # end for
            xs = np.array(xsl)
            ys = np.array(ysl)
            # perform regression

            slope, intercept = np.polyfit(np.log10(xs), np.log10(ys), 1)
            yfit = 10 ** (intercept + slope * np.log10(xs))
            # not necessary r_sq = r_squared(ys, ideal_y)
            print("<%d> | slope:%f" % (i, slope))
            axs.plot(xs, yfit, color='r')

        lines += line
        i += 1
    # end for

    # set limits
    for i in range(0, len(ax)):
        # ax.xaxis.grid(True, 'major')
        vertices = [(Xmin[i], Ymin[i]), (Xmax[i], Ymax[i])]

        ax[i].xaxis.grid(grid, 'minor')
        ax[i].yaxis.grid(grid, 'minor')
        # plt.setp(ax[i].get_xticklabels(), visible = True, fontsize = fontsize - 1)
        # plt.setp(ax[i].get_yticklabels(), visible = True, fontsize = fontsize - 1)
        ax[i].tick_params(axis='x', labelsize=fontsize - 8)
        ax[i].tick_params(axis='y', labelsize=fontsize - 8)
        # ax[i].set_yticklabels(fontsize=12)
        ax[i].grid(grid)
        ax[i].set_ylabel(Y_labels[i], fontsize=fontsize - 6)
        ax[i].set_xlabel(xlabel, fontsize=fontsize - 6)
        # fontd = {'family' : 'serif',
        #         'color'  : 'darkred',
        #         'weight' : 'normal',
        #         'size'   : 'large',
        # }
        # ax[i].yaxis.set_label_text(ylabel, fontdict=fontd)
        if i == 0:
            side = 'left'
        else:
            side = "right"

        if twoaxes or threeaxes:
            ax[i].axis[side].label.set_fontsize(fontsize)
            ax[i].axis["bottom"].label.set_fontsize(fontsize)

    plt.xlim(xmin=np.min(Xmin), xmax=np.max(Xmax))
    # this causes some problems
    # plt.ylim(ymin=np.min(Ymin), ymax=np.max(Ymax))

    if plottitle:
        plt.title(title, fontsize=fontsize)

    if legend is not None:
        labs = [l.get_label() for l in lines]
        # loc=4      position lower right
        # ---------------------------------------------------------- upper right    1
        # ---------------------------------------------------------- upper left    2
        # ---------------------------------------------------------- lower left    3
        # ---------------------------------------------------------- lower right    4
        # ---------------------------------------------------------- right    5
        # ------------------------------------------------------ --- center left    6
        # ---------------------------------------------------------- center right    7
        # ---------------------------------------------------------- lower center    8
        # ---------------------------------------------------------- upper center    9
        legnd = ax[0].legend(lines, labs, loc=3, frameon=False)
        for label in legnd.get_texts():
            label.set_fontsize(fontsize - 8)
        #plt.legend(frameon=False)
    plt.show()



def plotTimeSeries(title, xlabel, ylabel, x, y, legend = None, linewidth = 0.6, plottitle = False, \
                    doy = False, grid = False):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)

    if doy:
        dofy = timestamp2doy(x)
        ax.plot(dofy, y, linewidth = 0.6)
    else:
        ax.plot(x, y, linewidth = 0.6)

        # format the ticks
        formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(mondays)
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

    ax.xaxis.grid(grid, 'minor')
    ax.grid(grid)
    title = title

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if plottitle:
        plt.title(title)
    if legend != None:
        plt.legend(legend);

    plt.draw()
    plt.show()
# end

def plot_n_TimeSeries(title, xlabel, ylabel, x_arr, y_arr, legend = None, linewidth = 0.8, plottitle = False, fontsize = 18, \
                       doy = False, minmax = None, grid = False, show = True, dateinterval=None):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    ax = fig.add_subplot(111)
    
     
    i = 0;
    plt.gca().set_color_cycle(['blue', 'red', 'yellow', 'aqua', 'cyan', 'black'])
    ls = ['-', '-', '--', ':', '-.', '-', '--', ':', '-.']
    for a in x_arr:
        x = x_arr[i]
        y = y_arr[i]
        if doy:
            dofy = timestamp2doy(x)
            mindim=min(len(x),len(y))
            if mindim < len(x) or mindim < len(y):
                print("** Adjusting len(X)=%d or len(Y)=%d" %(len(x), len(y))) 
                x=x[:mindim]
                y=y[:mindim]    
            ax.plot(dofy, y, ls[i])
        else:
            if dateinterval != None:
                xs, ys = select_interval_data(x, y, dateinterval)
                doy=False
                print ("After select interval len(x)=%d, len(y)=%d" % (len(xs), len(ys)))
                ax.plot(xs, ys, ls[i])
            else:
                ax.plot(x, y, ls[i])
            

        # debug print Time, DOY and value
        if doy:
            for j in range(0, len(x)):
                if dofy[j] > 187 and dofy[j] < 190 :
                    print("T=%f, DOY=%f, Val=%f" % (x[j], dofy[j], y[j]))
        print("------------------------------------------------------")
        i += 1
    # end for
    if not doy and dateinterval ==None: 
        # format the ticks
        formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(mondays)
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()
    else:
        #If you want to avoid scientific notation in general,
        ax.ticklabel_format(useOffset=False, style='plain')
    
    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(grid, 'minor')
    ax.grid(grid)
    plt.ylabel(ylabel).set_fontsize(fontsize + 1)
    plt.xlabel(xlabel).set_fontsize(fontsize + 1)
    if plottitle:
        plt.title(title).set_fontsize(fontsize + 1)

    if legend != None:
        leg = plt.legend(legend)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    plt.xticks(fontsize = fontsize - 4)
    plt.yticks(fontsize = fontsize - 4)
    if minmax != None:
        plt.ylim(ymin = minmax[0], ymax = minmax[1])
    if show:
        plt.show()
# end

def flattop(N):
    a0 = 0.2810639
    a1 = 0.5208972
    a2 = 0.1980399
    w = np.zeros(N)
    for i in range(0, N):
        w[i] = a0 - a1 * math.cos(2 * np.pi * i / N) + a2 * math.cos(4 * np.pi * i / N)
    # end for
    return w
# end flattop

def findPeaks(self, f, mx):
    '''
    find max peaks
    '''
    # a = np.array([10.3, 2, 0.9, 4, 5, 6, 7, 34, 2, 5, 25, 3, -26, -20, -29], dtype = np.float)

    gradients = np.diff(mx)
    print(gradients)


    maxima_num = 0
    minima_num = 0
    max_locations = []
    min_locations = []
    count = 0
    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) & (i != gradients[count])):
            maxima_num += 1
            max_locations.append(count)

        if ((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) & (i != gradients[count])):
            minima_num += 1
            min_locations.append(count)
    # end for

    turning_points = {'maxima_number':maxima_num, 'minima_number':minima_num, 'maxima_locations':max_locations, 'minima_locations':min_locations}

    pass
    return turning_points
    # print turning_points

    # plt.plot(a)
    # plt.show()

def bandSumCentredOnPeaks(f, mx, band):
    '''
    find max peaks and band around it 'n' bins
    '''
    # turning_points = fft_utils.findPeaks(f, mx)
    # locations = turning_points['maxima_locations']
    # number = turning_points['maxima_number']

    # this works better
    if 1 == 0:
        xs = np.arange(0, np.pi, 0.05)
        data = np.sin(xs)
        peakind = sp.signal.find_peaks_cwt(data, np.arange(1, 10),
                                           wavelet = ricker , max_distances = None,
                                           gap_thresh = None, min_length = None,
                                           min_snr = 2.0, noise_perc = 10)
        print(peakind, xs[peakind], data[peakind])
        plt.plot(xs, data)
    # endif

    locations = sp.signal.find_peaks_cwt(mx, np.arange(1, 2),
                                         wavelet = scipy.signal.wavelets.ricker, max_distances = None,
                                         gap_thresh = 1, min_length = 1,
                                         min_snr = 1.0, noise_perc = 25)

    pd3 = peakdek.peakdet(mx, 1)


    for loc in locations:
        for b in range(-band / 2 + 1 , band / 2 + 1):
            if b != 0:
                mx[loc] += mx[loc + b].copy()
                mx[loc + b] = 0

    return mx






def RMS_values(data, nwaves):
    # detrend
    # demean

    rms = np.sqrt(1 / nwaves)

def autocorrelation(data, dt):
    '''This function calculates the autocorrelation function for the
       data set "origdata" for a lag timescale of 0 to "endlag" and outputs
       the autocorrelation function in to "a".
       function a = auto( origdata, endlag);
       (c) Dennis L. Hartmann
    '''

    N = len(data)
    a = np.zeros(N)
    endlag = N
    # now solve for autocorrelation for time lags from zero to endlag
    for lag in range(0, endlag - 1):
        data1 = data[0:N - lag]
        data1 = data1 - np.mean(data1)
        data2 = data[lag:N]
        data2 = data2 - np.mean(data2)
        print(lag)
        a[lag] = np.sum(data1 * data2) / np.sqrt(np.sum(data1 ** 2) * np.sum(data2 ** 2))
    # end for
    return a


def edof_stat(data, dt):
    # autocorrelation function
    a = autocorrelation(data, dt)
    n = len(data)
    Te = -dt / a
    tau = n * dt
    rt = math.exp(-tau / Te)
    dof = n * (-0.5 * math.log(rt))
    return dof


def dof(freq):
    '''
    NOTE: the confidence interval is calcuate for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================
    For non segmented data has 2 degrees of freedom for each datapoint Sj (Percival 1993 pag 20-21)
        Since Aj and Bj rv's are mutually uncorrelated on the assumption of Gaussianity for Aj and Bj

        and
         Data Analysis Methods in Physical Oceanography ( Emery 2001 ) p 424


        dof Sj is only a two degrees of freedom estimate of Sigma, there should be considerable variability Sj function
        '''
    dof = len(freq) - 2  # was = 2 - for each point of the FFT
    return dof


def edof(data, N, M, window_type):
    '''
    NOTE: the confidence interval is calcuate for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================

    Formulas from Prisley 1981:
    window_Type can be one of:
    ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'flattop']

    if window_type == 'flat':  # rectangular/Daniell
        dof = 1.0 * N / M
    if window_type == 'hanning':
        dof = (8.0 * N) / (3.0 * M)
    elif window_type == 'hamming':
        dof = 2.5164 * N / M
    elif window_type == 'bartlett':
        dof = 3.0 * N / M
    elif window_type == 'blackman':  # Not so sure about this one
        dof = 2.0 * N / M
    elif window_type == 'flattop':
        # ageneric formula or 50% ovelapping windows from  Data Analysis Methods in Physical Oceanography
        dof = 4.0 * N / M  #
    else:
         raise ValueError, "%f window not supported" % window_type
    '''

    '''
        Method from Spectral Analysis for Physical Applications  Percival 1993 p294
        Can be applied for 50% overlapping with Hanning windowing
        Other windows can be supported and other percents of overlapping
    '''
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'flattop']
    # number of blocks for a 50% overlap  N/M is the number of distinct segments
    nb = 2 * N / M - 1

    if not window_type in windows:
        raise "Window %s not supported" % window_type
    dof = 36 * nb * nb / (19 * nb - 1)


    return dof

def confidence_interval(data, dof, p_val, log = 'linear'):
    '''
    NOTE: the confidence interval is calculated for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================

    a() and b() are the 0.025 and 0.975% points of the X^2 distribution with v
    equivalent degrees of freedom [Priestley,1981, pp. 467-468].

    @param data : a time series or a FFT transform that is going to be evaluated.
                It can be a Power Density Spectrum ( module of Amplitude spectrum)
    @param dof  :Degrees of Freedom , usualy dof = n - scale for wavelets or n-2 for FFT

    @param p_val : the desired significance  p=0.05 for 95% confidence interval

    @return:  tuple of array with the interval  limits (a, b)
    '''

    p = 1 - p_val
    p_val = p / 2.

    chia = (stats.chi2.ppf(p_val, dof))  # typically 0.025
    a = dof / chia

    chib = (stats.chi2.ppf(1 - p_val, dof))  # typically 0.975
    b = dof / chib

    #if log != "loglog": even log log needs that
    a *= data
    b *= data

    # alternate calculations
    # ci = 1. / [(1 - 2. / (9 * dof) + 1.96 * math.sqrt(2. / (9 * dof))) ** 3, (1 - 2. / (9 * dof) - 1.96 * math.sqrt(2. / (9 * dof))) ** 3]

    return (b, a)



if __name__ == '__main__':

   
    ############################
    # filtering
    ###########################
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = filters.butter_bandpass(lowcut, highcut, fs, order = order)
        w, h = sp.signal.freqz(b, a, worN = 2000)
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

    y, w, h, N, delay = filters.butterworth(x, 'band', lowcut, highcut, fs, order = 6)
    if len(y) != len(t):
        t = scipy.signal.resample(t, len(y))
    plt.plot(t, y, label = 'Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles = '--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc = 'upper left')

    plt.show()

    ###################################
    # known signals on ufft
    ###################################
    fig = plt.figure(2)
    samp_rate = 200
    sim_time = 5
    freqs = [3., 5., 10., 40.]
    lowcut = 2.1
    highcut = 4.0

    nsamps = samp_rate * sim_time

    # generate input signal
    t = np.linspace(0, sim_time, nsamps)

    x = 0
    for i in range(len(freqs)):
        x += np.cos(2 * math.pi * freqs[i] * t)
    time_dom = fig.add_subplot(232)
    plt.plot(t, x)
    plt.title('Filter Input - Time Domain')
    plt.grid(True)

     # input signal spectrum
    xfreq = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(nsamps, d = 1. / samp_rate)
    fig.add_subplot(233)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(xfreq)[0:nsamps / 2])
    plt.title('Filter Input - Frequency Domain')
    plt.text(0.03, 0.01, "freqs: " + str(freqs) + " Hz")
    plt.grid(True)
    N = 5
    # b, a = filters.butter_bandpass(lowcut, highcut, samp_rate, order = N)
    # w, h = sp.signal.freqz(b, a, worN = 2000)
    # y = filters.butter_bandpass_filter(x, lowcut, highcut, samp_rate, order = N)
    btype = 'band'
    # btype = 'high'
    # btype = 'low'

    y, w, h, b, a = filters.butterworth(x, btype, lowcut, highcut, samp_rate)
    # y1 = filters.fft_bandpassfilter(x, samp_rate, lowcut, highcut)

    # filtered output
    fig.add_subplot(235)
    plt.title('Filter output - Time Domain')
    plt.grid(True)

    # output spectrum
    yfreq = np.fft.fft(y)
    fig.add_subplot(236)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(yfreq)[0:nsamps / 2])
    plt.title('Filter Output - Frequency Domain')
    plt.grid(True)

    debug = True
    if debug == True:
        fig_B = plt.figure()
        # plt.loglog(w * samp_rate / math.pi, np.abs(h))
        # w is in rad/sample therefore we ndde to multiply by sample and divide by pi
        plt.plot(w * samp_rate / (2 * math.pi), np.abs(h))
        plt.title('Filter Frequency Response')
        plt.text(2e-3, 1e-5, str(N) + "-th order " + 'band' + "pass Butterworth filter")
        plt.grid(True)

    plt.show()
