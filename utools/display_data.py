import numpy
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.pyplot.switch_backend('QT4Agg')
from datetime import datetime
from matplotlib.dates import MONDAY, SATURDAY
import matplotlib.dates as dates
import time, os, datetime, math, sys
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import zero_to_nan as z2n

import ufft.fft_utils as fft_utils
import ufft.filters as filters

months = dates.MonthLocator()  # every month
yearsFmt = dates.DateFormatter('%Y')
# every monday
mondays = dates.WeekdayLocator(MONDAY)

hour = dates.HourLocator(byhour = None, interval = 6, tz = None)



# plot x,y output of provec
def plot_hodograph(name, ur, vr, bins, dt, tunit = "sec", vunit = "m/s", disp_lunit = "m", fontsize = 20, title = False, modd = 9):
    '''
    @param name:the name of the location
    @param ur: East velocity vector
    @param vr: North velocity vector
    @param dt: time interval between velocity values
    @param tunit: time unit fot dt
    @param vunit: velocity unit
    @param disp_lunit: length unit on the final plot for X and Y
    '''

    quiv = True
    modd = modd
    scale = None
    if quiv:
        xqa = []
        yqa = []
        uqa = []
        vqa = []

    def build_progressive_vector(ur, vr, bins, dt, tunit = tunit, vunit = vunit):
        '''
        We build here a vector with distances in disp_lunit X and  Y
        The intitial vleocities are transformed in m/s  (SI) if not original so

        dt  - delta t in days
        '''

        xa = []
        ya = []
        legend = []

        if tunit == "sec":
            dt *= 86400
            if vunit == "m/s":
                vfac = 1.

            elif vunit == "Km/h":
                vfac = 1000. / 3600.
            elif vunit == "cm/s":
                vfac = 1. / 100.
        elif tunit == "min":
            dt *= 86400 / 60.
            if vunit == "m/s":
                vfac = 60.
            elif vunit == "Km/h":
                vfac = 60.*1000. / 3600.
            elif vunit == "cm/s":
                vfac = 60. / 100.
        elif tunit == "hour":
            dt *= 24
            if vunit == "m/s":
                vfac = 3600.
            elif vunit == "Km/h":
                vfac = 1000.
            elif vunit == "cm/s":
                vfac = 3660. / 100.


        for j in range(0, len(bins)):
            x = [0]  # start from the origin
            y = [0]
            if quiv:
                xq = []
                yq = []
                uq = []
                vq = []

            for i in range(1, len(ur[j])):
                x.append(x[i - 1] + (ur[j][i] + ur[j][i - 1]) / 2 * vfac * dt)
                y.append(y[i - 1] + (vr[j][i] + vr[j][i - 1]) / 2 * vfac * dt)
                if quiv :
                    xq.append(x[i - 1])
                    yq.append(y[i - 1])
                    uq.append(x[i] - x[i - 1])
                    vq.append(y[i] - y[i - 1])

            if quiv:
                xqa.append(xq[::modd])
                yqa.append(yq[::modd])
                uqa.append(uq[::modd])
                vqa.append(vq[::modd])

            xa.append(x)
            ya.append(y)

            legend.append("bin " + str(bins[j]))
        return xa, ya, legend
    # end build_progressive_vector

    def print_arrows(ax, xqa, yqa, uqa, vqa):
        for bin in range(0, len(xqa)):
            xq = xqa[bin]
            yq = yqa[bin]
            uq = uqa[bin]
            vq = vqa[bin]
            for i in range(0, len(xq) - 1):
                # arr = plt.Arrow(xq[i], yq[i], uq[i], vq[i], width = 2000.0, color = 'k', fc = 'black', ls = 'dotted', visible = True, zorder = 100)
                # plt.gca().add_patch(arr)
                ax.arrow(xq[i], yq[i], uq[i], vq[i], shape = 'full', lw = 3.5, length_includes_head = True, head_width = 20.0, zorder = 100)
    # end print arrows
    xa, ya, legend = build_progressive_vector(ur, vr, bins, dt, tunit = tunit, vunit = vunit)

    fig = plt.figure()
    fig.canvas.set_window_title(name)
    ax = fig.add_subplot(111)
    pen1 = ["c-", "r-", "y-"]
    pen2 = ["go", "go", "go"]
    pen3 = ["rd", "rd", "rd"]
    for i in range(0, len(xa)):
        ur = xa[i]
        vr = ya[i]
        ax.plot(ur, vr, pen1[i], lw = 3.8, label = legend[i])

        ax.plot(ur[0], vr[0], pen2[i], markersize = 12, zorder = 200)
        ax.plot(ur[-1], vr[-1], pen3[i], markersize = 12, zorder = 200)

    if quiv:
        print(xqa[0])
        print(yqa[0])
        print(uqa[0])
        print(vqa[0])

        # ax.quiver(xqa, yqa, uqa, vqa, angles = "xy", scale_units = "xy", width = 0.003, headwidth = 4, scale = scale)
        print_arrows(ax, xqa, yqa, uqa, vqa)
    ax.set_aspect('equal')
    ax.autoscale(tight = True)
    ax.set_xlabel('Displacement west-east [' + disp_lunit + ']').set_fontsize(fontsize)
    ax.set_ylabel('Displacement north-south [' + disp_lunit + ']').set_fontsize(fontsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    if title:
        plt.title(name).set_fontsize(fontsize + 2)
    plt.show()

def normalize_ts(timeseries):
    normal=(timeseries - timeseries.min()) / (timeseries.max() - timeseries.min())
    return  normal

def display_scatter(x, y, slope = False, type = 'stats', labels = None, fontsize = 20):
    if type == 'stats':
        # or with stats
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        if slope:
            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
            print("R-value=%f" % r_value)
            # Calculate some additional outputs
            predict_y = intercept + slope * x
            pred_error = y - predict_y
            degrees_of_freedom = len(x) - 2
            residual_std_error = numpy.sqrt(numpy.sum(pred_error ** 2) / degrees_of_freedom)
            # Plotting
            leg = "R$^2$=%0.2f" % (r_value+0.05)
            ax.annotate(leg, xy=(85, 320), xycoords='axes points',
                        size=14, ha='right', va='top',
                        bbox=dict(boxstyle='round', edgecolor='white', fc='w'))
            ax.plot(x, predict_y, 'k-')

    elif type == 'simple':
        plt.scatter(x, y, marker = "square", color = "blue")
        if slope:
            # perform regression
            slope, intercept = np.polyfit(x, y, 1)
            yfit = intercept + slope * x
            # not necessary r_sq = r_squared(ys, ideal_y)
            # print "<%d> | slope:%f" % (i, slope)
            plt.plot(x, yfit, color = 'r')
    elif type == 'linalg':

        if slope:
            w = linalg.lstsq(x.T, y)[0]  # obtaining the parameters
            # plotting the line
            line = w[0] * x + w[1]  # regression line
            plt.plot(x, line, 'r-')

    if labels != None:
         plt.xlabel(labels[0]).set_fontsize(fontsize)
         plt.ylabel(labels[1]).set_fontsize(fontsize) 
    
    plt.xticks(fontsize = fontsize - 2)     
    plt.yticks(fontsize = fontsize - 2)
    
    plt.show()

def display(dateTime, temp, label, k):
    fig = plt.figure(num = k, facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    ax.plot(dateTime, temp, linewidth = 1.6, color = 'r')
    # format the ticks
    formatter = dates.DateFormatter('%Y-%m')
    # formatter = dates.DateFormatter('`%y')
    # ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mondays)

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    title = ' Station %d' % k
    ylabel = ' temperature ($^\circ$C)'
    plt.ylabel(ylabel).set_fontsize(20)
    plt.title(title).set_fontsize(22)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


def display2(dateTime, temp, coeff, k):
    fig = plt.figure(num = k, facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    ax.plot(dateTime, temp, 'r', dateTime, coeff, 'b', linewidth = 1.4)
    # format the ticks
    formatter = dates.DateFormatter('%Y-%m-%d')
    # formatter = dates.DateFormatter('`%y')
    # ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mondays)

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    title = ' Station %d' % k
    ylabel = ' Temperature [$^\circ$C]'
    plt.ylabel(ylabel).set_fontsize(20)
    plt.title(title).set_fontsize(22)
    plt.show()

def display_upwelling(dateTimes, temps, coeffs, k):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    i = 0
    legend = []
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    for dateTime in dateTimes:
        temp = temps[i]
        coef = coeffs[i]
        ax.plot(dateTime, coef, linestyle = ls[i], linewidth = 1.2 + 0.2 * i)
        # ax.plot(dateTime, temp, linewidth = 0.6)
        lg = 'Station %d' % k[i]
        legend.append(lg)
        i += 1

    # format the ticks
    formatter = dates.DateFormatter('%Y-%m-%d')
    # formatter = dates.DateFormatter('`%y')
    # ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(hour)

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    ylabel = ' Temperature [$^\circ$C]'
    plt.ylabel(ylabel).set_fontsize(20)
    title = ' Upwelling advancement'

    plt.title(title).set_fontsize(22)
    plt.legend(legend)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()

def display_temperature(dateTimes, temps, coeffs, k, fnames = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    i = 0
    legend = []
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    for dateTime in dateTimes:
        temp = temps[i]
        coef = coeffs[i]
        ax.plot(dateTime, coef, linestyle = ls[i], linewidth = 1.2 + 0.2 * i)
        # ax.plot(dateTime, temp, linewidth = 0.6)
        if fnames == None:
            lg = 'Sensor %s' % k[i][1]
        else:
            lg = '%s' % fnames[i]
        legend.append(lg)
        i += 1

    # format the ticks
    formatter = dates.DateFormatter('%Y-%m-%d')
    # formatter = dates.DateFormatter('`%y')
    # ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(formatter)
    # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(hour)

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    ylabel = ' Temperature [$^\circ$C]'
    plt.ylabel(ylabel).set_fontsize(20)
    title = ' Temperature Profiles'

    plt.title(title)
    plt.legend(legend)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


def display_one_temperature(dateTime, temps, leg=None, title=None, doy=False, grid=False, fontsize=20, ylabel=None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    i = 0
    legend = []
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    if doy:
        dT = fft_utils.timestamp2doy(dateTime)
    else:
        dT = dateTime[:]
    ax.plot(dT, temps, linestyle = ls[i], linewidth = 1.2 + 0.2 * i)
    legend.append(legend)


    if doy:
        xlabel = "Day of Year"
    else:
        xlabel = "Time"
        # format the ticks
        formatter = dates.DateFormatter('%Y-%m-%d')
        # formatter = dates.DateFormatter('`%y')
        # ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        #ax.xaxis.set_minor_locator(hour)
        # ax.xaxis.grid(True, 'major')
        #ax.xaxis.grid(True, 'minor')
        # axes up to make room for them
        fig.autofmt_xdate()

    ax.grid(grid)
    if ylabel is None:
        ylabel = ' Temperature [$^\circ$C]'
    plt.ylabel(ylabel).set_fontsize(fontsize)
    plt.xlabel(xlabel).set_fontsize(fontsize)
    plt.xticks(fontsize = fontsize - 4)
    plt.yticks(fontsize = fontsize - 4)
    if title is not None:
        plt.title(title)
    if leg is not None:
        plt.legend(leg)

    # rotates and right aligns the x labels, and moves the bottom of the
    plt.show()


def display_temperatures_and_peaks(dateTimes, temps, maxpeaks, minpeaks, k, fnames = None, revert = False, difflines = False, \
                                      custom = None, maxdepth = None, tick = None, firstlog = None, fontsize = 20, ylim = None, \
                                      fill = False, show = True, minorgrid = None, datetype = 'date'):
    '''
    '''

    if True:
        ax = display_temperatures(dateTimes, temps, k, fnames = fnames, revert = revert, difflines = difflines, custom = custom, \
                              maxdepth = maxdepth, tick = tick, firstlog = firstlog, fontsize = fontsize, ylim = ylim, fill = fill, \
                               show = False, minorgrid = minorgrid, datetype = datetype)
    else:
         fig = plt.figure(facecolor = 'w', edgecolor = 'k')
         ax = fig.add_subplot(111)
    for i in range(len(maxpeaks)):
        xm = [p[0] for p in maxpeaks[i]]
        ym = [p[1] for p in maxpeaks[i]]
        xn = [p[0] for p in minpeaks[i]]
        yn = [p[1] for p in minpeaks[i]]

        # plot local min and max
        ax.plot(xm, ym, 'ro')
        ax.plot(xn, yn, 'bo')

    if show:
        plt.show()
# end display_temperatures_peaks


def display_temperatures(dateTimes, temps, k, fnames = None, revert = False, difflines = False, custom = None,
                         maxdepth = None, tick = None, firstlog = None, fontsize = 20, ylim = None, fill = False,
                         show = True, datetype = 'date', minorgrid = None, grid = None, ylab = None, settitle = False,
                          draw_xaxis = False, legendloc='upper right'):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)
    i = 0
    legend = []
    title = None

    ls = ['-', '-', '-', '-', '-', '-', '-', '-.', '-', '-', ':', '-.', '-', '--', ':', '-.']
    colour_list = ['b', 'r', 'y', 'g', 'c', 'm', 'k',
                   'brown', 'darkmagenta', 'cornflowerblue', 'darkorchid', 'crimson', 'darkolivegreen', 'chartreuse']
    #===========================================================================
    # colour_list = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
    #                'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    #                'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki',
    #                'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid']
    #===========================================================================
    miny = 1e10
    maxy = -1e10
    for dT in dateTimes:
        temp = temps[i]
        if revert == True:
            reversed_temp = temp[::-1]
        else:
            reversed_temp = temp

        if datetype == 'dayofyear':
            dateTime = fft_utils.timestamp2doy(dT[1:])
        else:
            dateTime = dT[1:]
        if difflines:
            ax.plot(dateTime, reversed_temp[1:], linestyle = ls[i], linewidth = 1.6 + 0.1 * i, color = colour_list[i])
        else:
            try:
                if len(dateTime) > 0:
                    ax.plot(dateTime, reversed_temp[1:], linewidth = 1.1, color = colour_list[i])
            except Exception as e:
                print("Error %s" % e)
                continue

        if fnames is not None:
            if isinstance(fnames, str):
                if fnames.rfind('.') == -1:
                    lg = "%s" % (fnames)
                else:
                    fileName, fileExtension = os.path.splitext(fnames)
                    if fileName[0:3] == "zz_":
                        fileName = fileName[3:]
                    lg = '%s' % fileName
            elif isinstance(fnames, list) or isinstance(fnames, numpy.ndarray):
                if fnames[i].rfind('.') == -1:
                    lg = "%s" % (fnames[i])
                else:
                    fileName, fileExtension = os.path.splitext(fnames[i])
                    if fileName[0:3] == "zz_":
                        fileName = fileName[3:]
                    lg = '%s' % fileName

        
            legend.append(lg)
        ax.set_xlim(xmax = dateTime[len(dateTime) - 1])

        i += 1
        if draw_xaxis:
            miny = min(numpy.min(temp), miny)
            maxy = max(numpy.max(temp), maxy)
            minx = dateTime[0]
            maxx = dateTime[len(dateTime) - 1]
    # end for
    if fill == True and len(dateTimes) == 2 and len(dateTimes[1]) == len(dateTimes[0]):
        sd = 0.5
        ax.fill_between(dateTimes[0], temps[1], temps[0], where = temps[1] <= temps[0], facecolor = [sd, sd, sd], interpolate = True)

    # format the ticks
    if datetype == 'date':
        formatter = dates.DateFormatter('%Y-%m-%d')
        # formatter = dates.DateFormatter('`%y')
        # ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))

        if minorgrid != None and grid != None:
            if minorgrid == 'hour':
                mgrid = hour
            elif minorgrid == 'mondays':
                mgrid = mondays
            else:
                mgrid = None
            if mgrid != None:
                ax.xaxis.set_minor_locator(mgrid)
        fig.autofmt_xdate()

    plt.xticks(fontsize = fontsize - 2)
    plt.xlabel("Day of year").set_fontsize(fontsize + 2)


    # ax.xaxis.set_minor_locator(mondays)

    # ax.xaxis.grid(True, 'major')
    if grid != None:
        ax.xaxis.grid(True, 'minor')
        ax.grid(True)

    if tick != None:
       ax.set_yticks(tick[1])
       ax.set_yticklabels(tick[0])

    if custom == None:
        ylabel = ' Temp. ($^\circ$C)'
        title = ' Temperature Profiles'

    else:
        # title = ' Profiles: %s' % custom
        title = ' %s' % custom
        if ylab == None:
            ylabel = title
        else:
            ylabel = ylab

    if ylim != None:
         ax.set_ylim(ylim[0], ylim[1])

    sign = lambda x: math.copysign(1, x)
    if draw_xaxis:
        if sign(miny) != sign(maxy):
            plt.plot([minx, maxx], [0, 0], '--', color = 'k', linewidth = 1.2)


    plt.ylabel(ylabel).set_fontsize(fontsize + 2)
    plt.yticks(fontsize = fontsize - 2)

    # labels = ax.get_xticklabels()
    # plt.setp(labels, rotation = 0, fontsize = fontsize - 2)

    if settitle:
        plt.title(title).set_fontsize(fontsize + 2)
    if fnames is not None:
        plt.legend(legend, fontsize = fontsize - 6, loc=legendloc)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them

    # avoid overlapping of labels at origin
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune = 'lower'))


    if show:
        plt.show()
    return ax


def display_marker_histogram(xarr, yarr, fnamesarr, xlabel, ylabel, title = None, log = False, grid = True, fontsize = 18):
    '''
    Display the array as histogram with markers.
    '''

    format = 100 * len(xarr) + 10

    params = {'legend.fontsize': fontsize - 6}
    plt.rcParams.update(params)

    lenght = len(xarr)
    ax = numpy.zeros(lenght, dtype = matplotlib.axes.Subplot)
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    marker = ['o', '*', '^', 'D', 's', 'p', 'o', '+', 'x', 'H']
    #colour = ['MediumTurquoise', 'Chartreuse', 'Yellow', 'Fuchsia', 'k', 'b', 'aqua', 'r', 'g', 'k', 'm', 'c', 'y']
    colour = ['black','blue','red','limegreen', 'darkorange','darkturquoise','teal','magenta','dodgerblue','firebrick','olivedrab']


    for j in range(0, lenght):
        legend = []
        if j == 0 :
            ax[j] = fig.add_subplot(format + j + 1)
        else:
            ax[j] = fig.add_subplot(format + j + 1, sharex = ax[0])

        fnames = fnamesarr[j]
        x = xarr[j]
        y = yarr[j]
        for i in range(0, len(x)):
            if log:
                plt.yscale('log')

            # Filter out
            yvalues = z2n.zero_to_nan(y[i])
            ax[j].plot(x[i], yvalues, marker = marker[i], markersize = 9, lw = 1.4, color = colour[i], \
                      markerfacecolor = 'None', markeredgecolor = colour[i])

            # Plot legend
            if fnames is None:
                legend.append("Sensor %s" % k[i][1])
            else:
                if fnames[i].rfind('.') == -1:
                    legend.append("%s" % (fnames[i]))
                else:
                    fileName, fileExtension = os.path.splitext(fnames[i])
                    legend.append('%s' % fileName)

        # end for i
        ax[j].legend(legend)
        # Set the fontsize
        # for label in plt.legend().get_texts():
        #    label.set_fontsize('large')

        if title != None:
            plt.title(title).set_fontsize(fontsize + 2)
        if j == lenght - 1:
            ax[j].set_xlabel(xlabel).set_fontsize(fontsize)
        ax[j].set_ylabel(ylabel).set_fontsize(fontsize + 2)
        for tick in ax[j].xaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize - 4)
        for tick in ax[j].yaxis.get_major_ticks():
                tick.label.set_fontsize(fontsize - 4)
        ax[j].grid(grid, axis = 'both')


    # end for j
    plt.show()


def display_temperatures_subplot(dateTimes, temps, coeffs, k, fnames = None, revert = False, custom = None, maxdepth = None, tick = None, \
                                 firstlog = None, yday = None, delay = None, group = None, title = False, grid = False, processed = False, \
                                 limits = None, sharex = False):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    if group != None:
        format = 100 * (len(dateTimes) / 2) + 10
    else:
        format = 100 * len(dateTimes) + 10
    matplotlib.rcParams['legend.fancybox'] = True

    # if delay != None and yday == None:
    #    raise BasicException("delay needs yday=True")



    # ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    if yday != None and delay != None:
        minx = 10000000.
        maxx = 0.

        # find maxx and minX of the X axis
        for k in range(0, len(dateTimes)):
            d = dateTimes[k]
            dmax = dates.num2date(d[len(d) - 1])
            maxx = max(maxx, (dmax.timetuple().tm_yday + dmax.timetuple().tm_hour / 24. + dmax.timetuple().tm_min / (24. * 60) + dmax.timetuple().tm_sec / (24. * 3600)))

            dmin = dates.num2date(d[1])
            minx = min(minx, (dmin.timetuple().tm_yday + dmin.timetuple().tm_hour / 24. + dmin.timetuple().tm_min / (24. * 60) + dmin.timetuple().tm_sec / (24. * 3600) - delay[k]))


    i = 0
    if group != None:
        lenght = len(dateTimes) / 2
    else:
        lenght = len(dateTimes)
    ax = numpy.zeros(lenght, dtype = matplotlib.axes.Subplot)

    for j in range(0, lenght):
        if (sharex and j == 0) or not sharex:
            ax[j] = fig.add_subplot(format + j + 1)
        else:
            ax[j] = fig.add_subplot(format + j + 1, sharex = ax[0])

        if group != None:
            temp = [temps[i][1:], temps[i + 1][1:]]
            coef = [coeffs[i][1:], coeffs[i + 1][1:]]
            dateTime = [dateTimes[i][1:], dateTimes[i + 1][1:]]
        else:
            temp = temps[i][1:]
            coef = coeffs[i][1:]
            dateTime = dateTimes[i][1:]

        # ax.plot(dateTime[1:], coef[1:])
        if revert == True:
            if group != None:
                reversed_temp = [temps[i][::-1], temps[i + 1][::-1]]
                reversed_coef = [coeffs[i][::-1], coeffs[i + 1][::-1]]
            else:
                reversed_temp = temps[i][::-1]\

                reversed_coef = coeffs[i][::-1]
        else:
            reversed_temp = temp
            reversed_coef = coef

        lone = ""
        ltwo = ""
        if fnames == None:
            lg = "Sensor %s" % k[i][1]
        else:
            if fnames[i].rfind('.') == -1:
                if group != None:
                    lone = "%s" % (fnames[i])
                    ltwo = "%s" % (fnames[i + 1])
                else:
                    lone = "%s" % (fnames[i])
            else:
                if group != None:
                    fileName1, fileExtension1 = os.path.splitext(fnames[i])
                    fileName2, fileExtension2 = os.path.splitext(fnames[i + 1])
                    lone = '%s' % fileName1
                    ltwo = '%s' % fileName2
                else:
                    fileName, fileExtension = os.path.splitext(fnames[i])
                    lone = '%s' % fileName

        if processed:
            pdata = reversed_coef
        else:
            pdata = reversed_temp


        if yday == None:
            if group != None:
                lplt1 = ax[j].plot(dateTime[0], pdata[0], linewidth = 1.2, color = 'r', label = lone)
                lplt2 = ax[j].plot(dateTime[1], pdata[1], linewidth = 1.2, color = 'b', label = ltwo)

            else:
                lplt = ax[j].plot(dateTime, pdata, linewidth = 1.2, label = lone)
            # end if group


        else:

            dely = delay[i] if delay != None else 0.0

            # dates = [datetime.fromordinal(d) for d in dataTime]
            # dofy = [d.tordinal() - datetime.date(d.year, 1, 1).toordinal() + 1 for d in dates]
            if group != None:
                dtime1 = dateTime[0]
                dtime2 = dateTime[1]
                dofy1 = numpy.zeros(len(dtime1))
                for k in range(0, len(dtime1)):
                    d1 = dates.num2date(dtime1[k])
                    dofy1[k] = d1.timetuple().tm_yday + d1.timetuple().tm_hour / 24. + d1.timetuple().tm_min / (24. * 60) + d1.timetuple().tm_sec / (24. * 3600) - dely
                # end for
                dofy2 = numpy.zeros(len(dtime2))
                for k in range(0, len(dtime2)):
                    d2 = dates.num2date(dtime2[k])
                    dofy2[k] = d2.timetuple().tm_yday + d2.timetuple().tm_hour / 24. + d2.timetuple().tm_min / (24. * 60) + d2.timetuple().tm_sec / (24. * 3600) - dely
                # end for
                d2 = dates.num2date(dtime2)
            else:
                dtime1 = dateTime
                dofy1 = numpy.zeros(len(dtime1))
                for k in range(0, len(dtime1)):
                    d1 = dates.num2date(dtime1[k])
                    dofy1[k] = d1.timetuple().tm_yday + d1.timetuple().tm_hour / 24. + d1.timetuple().tm_min / (24. * 60) + d1.timetuple().tm_sec / (24. * 3600) - dely
                # end for
            # end if

            if group != None:
                lplt1 = ax[j].plot(dofy1, pdata[0], linewidth = 1.2, color = 'r', label = lone)
                lplt2 = ax[j].plot(dofy2, pdata[1], linewidth = 1.2, color = 'b', label = ltwo)
            else:
                lplt = ax[j].plot(dofy1, pdata, linewidth = 1.2, label = lone)

        # LEGEND
        # blue_proxy = plt.Rectangle((0, 0), 1, 1, fc = "b")
        # ax[i].legend([blue_proxy], ['cars'])
        # ax[j].legend(shadow = True, fancybox = True)
        handles, labels = ax[j].get_legend_handles_labels()
        ax[j].legend(handles, labels)

        # X-AXIS -Time
        # format the ticks
        if yday == None:
            formatter = dates.DateFormatter('%Y-%m-%d')
            # formatter = dates.DateFormatter('`%y')

            ax[j].xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
            ax[j].xaxis.set_minor_locator(mondays)
            fig.autofmt_xdate()

        if grid:
            # ax.xaxis.grid(True, 'major')
            ax[j].yaxis.grid(True, 'minor')
        else:
            ax[j].yaxis.grid(False),
        ax[j].xaxis.grid(True, 'major')

        if custom == None:
            ylabel = ' T [$^\circ$C]'
            ax[j].set_ylabel(ylabel).set_fontsize(18)
            if title:
                title = ' Temperature Profiles - %s' % lone
        else:
            if title:
                title = ' Profile: %s' % custom[i]
                ax[j].set_ylabel(custom[i])
        if title:
            ax[j].set_title(title).set_fontsize(20)

        if yday == None:
            if group != None:
                ax[j].set_xlim(xmax = dateTime[0][len(dateTime) - 1])
                if j == lenght - 1:
                    ax[j].set_xlabel("Time").set_fontsize(20)
            else:
                ax[j].set_xlim(xmax = dateTime[len(dateTime) - 1])
                if j == lenght - 1:
                    ax[j].set_xlabel("Time").set_fontsize(20)
        else:
            if delay != None:
                ax[j].set_xlim(xmin = minx, xmax = maxx)

            if j == lenght - 1:
                ax[j].set_xlabel("Day of year").set_fontsize(20)

        # limits
        if limits != None:
            ax[j].set_ylim(ymin = limits[0] , ymax = limits[1])


        if maxdepth != None:
            if firstlog != None:
                mindepth = firstlog
            else:
                mindepth = 0
            ax[j].set_ylim(mindepth, maxdepth[i])

        # ax[j].legend(lplt, title = lg, shadow = True, fancybox = True)
        if tick != None:
           ax[j].set_yticks(tick[i][1])
           ax[j].set_yticklabels(tick[i][0])

        # set labels visibility
        plt.setp(ax[j].get_xticklabels(), visible = True)

        if group != None:
            i += 2
        else:
            i += 1

        if sharex and j < lenght - 1:
            plt.setp(ax[j].get_xticklabels(), visible = False)
        elif sharex and j == lenght - 1:
            plt.setp(ax[j].get_xticklabels(), visible = True)
    # end for

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()

    # new seting to make room for labels
    # plt.tight_layout()

    plt.draw()
    plt.show()


def display_depths_subplot(dateTimes, depths, maxdepth, fnames = None, yday = None, revert = True, tick = None, custom = None, firstlog = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    format = 100 * len(dateTimes) + 10
    matplotlib.rcParams['legend.fancybox'] = True


    if yday != None:
        minx = 10000000.
        maxx = 0.

        # find maxX and minX of the X axis
        for k in range(0, len(dateTimes)):
            d = dateTimes[k]
            dmax = dates.num2date(d[len(d) - 1])
            maxx = max(maxx, (dmax.timetuple().tm_yday + dmax.timetuple().tm_hour / 24. + dmax.timetuple().tm_min / (24. * 60) + dmax.timetuple().tm_sec / (24. * 3600)))

            dmin = dates.num2date(d[0])
            minx = min(minx, (dmin.timetuple().tm_yday + dmin.timetuple().tm_hour / 24. + dmin.timetuple().tm_min / (24. * 60) + dmin.timetuple().tm_sec / (24. * 3600)))


    i = 0
    # ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    ax = numpy.zeros(len(dateTimes), dtype = matplotlib.axes.Subplot)
    for dateTime in dateTimes:
        ax[i] = fig.add_subplot(format + i + 1)
        depth = depths[i]
        # ax.plot(dateTime[1:], coef[1:])
        if (revert == True and tick == None) or (revert == None and tick != None):
            raise IOError('Both revert and tick must be defined')

        if revert == True:
            reversed_depth = maxdepth[i] - depth
        else:
            reversed_depth = depth

        if fnames == None:
            lg = "Sensor %s" % k[i][1]
        else:
            if fnames[i].rfind('.') == -1:
                lg = "%s" % (fnames[i])
            else:
                fileName, fileExtension = os.path.splitext(fnames[i])
                lg = '%s' % fileName

        if yday == None:
            lplt = ax[i].plot(dateTime[1:], reversed_depth[1:], linewidth = 1.4, label = lg)
        else:
            dofy = numpy.zeros(len(dateTime))
            # dates = [datetime.fromordinal(d) for d in dataTime]
            # dofy = [d.tordinal() - datetime.date(d.year, 1, 1).toordinal() + 1 for d in dates]
            for j in range(0, len(dateTime)) :
                d = dates.num2date(dateTime[j])
                dofy[j] = d.timetuple().tm_yday + d.timetuple().tm_hour / 24. + d.timetuple().tm_min / (24. * 60) + d.timetuple().tm_sec / (24. * 3600)

            lplt = ax[i].plot(dofy[1:], reversed_depth[1:], linewidth = 1.4, label = lg)

        # LEGEND
        # blue_proxy = plt.Rectangle((0, 0), 1, 1, fc = "b")
        # ax[i].legend([blue_proxy], ['cars'])
        ax[i].legend(shadow = True, fancybox = True)


        # X-AXIS -Time
        # format the ticks
        if yday == None:
            formatter = dates.DateFormatter('%Y-%m-%d')
            # formatter = dates.DateFormatter('`%y')

            ax[i].xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
            ax[i].xaxis.set_minor_locator(mondays)


        # ax.xaxis.grid(True, 'major')
        ax[i].xaxis.grid(True, 'minor')
        ax[i].grid(True)
        if custom == None:
            ylabel = ' Depth. (m)'
            ax[i].set_ylabel(ylabel).set_fontsize(20)
            title = '  Profiles - %s' % lg
        else:
            title = ' Profile: %s' % custom[i]
            ax[i].set_ylabel(custom[i])

        ax[i].set_title(title).set_fontsize(22)
        if yday == None:
            ax[i].set_xlim(xmax = dateTime[len(dateTime) - 1])
            ax[i].set_xlabel("Time").set_fontsize(20)
        else:
            ax[i].set_xlim(xmin = minx, xmax = maxx)
            ax[i].set_xlabel("day of the year").set_fontsize(20)

        if maxdepth != None:
            if firstlog != None:
                mindepth = firstlog
            else:
                mindepth = 0
            ax[i].set_ylim(mindepth, maxdepth[i])

        # ax[i].legend(lplt, title = lg, shadow = True, fancybox = True)
        if tick != None:
           ax[i].set_yticks(tick[i][1])
           ax[i].set_yticklabels(tick[i][0])


        i += 1

    # end for

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them


    fig.autofmt_xdate()
    plt.draw()
    plt.show()



def display_mixed_subplot(dateTimes1=[], data=[] , varnames=[], ylabels1=[], limits1=[],
                          dateTimes2=[], groups=[], groupnames=[], ylabels2=[],
                          dateTimes3=[], imgs=[], ylabels3=[], ticks=[], maxdepths=[], firstlogs = [],
                          maxtemps=[], mindepths = [], mintemps = [],
                          interp = None, fnames = None, revert = False, custom = None, maxdepth = None,
                          tick = None, firstlog = None, yday=False,
                          title = False, grid = False, limits = None, sharex=False, sharey=False, fontsize = 18,
                          group_first = False, cblabel =  [None, None, None, None], label=None):
    '''
    dateTimes1 = [] - for single graphs
    dateTimes2 = [] - for pairs of data graphs
    dateTimes3 = [] - for images
    :param label: {str} default=None, one of 'fancy', 'frameless'
    '''

    def eval_group_order_o(loop, group_first, len1, len2, len3, j, il, ig, groups):
        if loop == 1:
            if group_first:
                return j > len3 - 1 and j < len2 + len3 and il < len2
            else:
                return j < len2 and il < len2
        if loop == 2:
            if group_first:
                return j < len3 and ig < len(groups)
            else:
                return j > len2 - 1 and j < len2 + len3 and ig < len(groups)
    # end eval_group_order_o

    def eval_group_order(loop, group_first, len1, len2, len3, j, il, ig, groups):

        if loop == 1 and j < len1:
            return True
        elif loop == 1 and j >= len1:
            return False
        else:
            if loop == 2 and j >= len1 and j < len1 + len2 :
                return True
            elif loop == 2 and j >= len1 and j >= len1 + len2:
                return False
            else:
                if loop == 3 and j >= len1 and j >= len1 + len2 and j < len1 + len2 +len3:
                    return True
                else:
                    return False
    # end eval_group_order

    colour_list = ['k', 'darkmagenta', 'g', 'b', 'r', 'y', 'g', 'c', 'm',
                   'brown', 'darkmagenta', 'cornflowerblue', 'darkorchid', 'crimson', 'darkolivegreen', 'chartreuse']

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    len1 = len(dateTimes1)
    len2 = len(dateTimes2) / 2
    len3 = len(dateTimes3)
    length = int(len1 + len2 + len3)
    format = 100 * length + 10
    if len(varnames) == len1 and label == 'fancy':
        matplotlib.rcParams['legend.fancybox'] = False # True
    if yday:
        datetype = 'dayofyear'
        minx = 10000000.
        maxx = 0.

        # find maxx and minX of the X axis
        for k in range(0, len(dateTimes1)):
            d = dateTimes1[k]
            dmax = dates.num2date(d[len(d) - 1])
            maxx = max(maxx, (dmax.timetuple().tm_yday + dmax.timetuple().tm_hour / 24. + dmax.timetuple().tm_min / (24. * 60) + dmax.timetuple().tm_sec / (24. * 3600)))

            dmin = dates.num2date(d[1])
            minx = min(minx, (dmin.timetuple().tm_yday + dmin.timetuple().tm_hour / 24. + dmin.timetuple().tm_min / (24. * 60) + dmin.timetuple().tm_sec / (24. * 3600)))
    else:
        datetype = 'date'
    # end if

    i = 0

    ax = numpy.zeros(length, dtype = matplotlib.axes.Subplot)
    axb = numpy.zeros(len3, dtype = matplotlib.axes.Subplot)
    il = 0  # index one line
    ig = 0  # index group
    ii = 0  # index img

    gs1 = gridspec.GridSpec(length, 1)

    for j in range(0, length):
        if sharey:
            drawylabel = False
        else:
            drawylabel = True
        if j == 0:
            ax[j] = fig.add_subplot(gs1[j])
        else:
            ax[j] = fig.add_subplot(gs1[j], sharex=ax[0])
        if eval_group_order(1, group_first, len1, len2, len3, j, il, ig, groups):
            if revert != True:
                temp = data[il][:]
                dateTime = dateTimes1[il][:]
            else:
                temp = data[il][::-1]
                dateTime = dateTimes1[il][::-1]

            if yday == False:
                if len(varnames) == len1:
                    lplt = ax[j].plot(dateTime, temp, linewidth = 1, label = varnames[il], color = colour_list[il])
                else:
                    lplt = ax[j].plot(dateTime, temp, linewidth=1, color=colour_list[il])
            else:
                dtime1 = dateTime
                dofy1 = numpy.zeros(len(dtime1))
                for k in range(0, len(dtime1)):
                    d1 = dates.num2date(dtime1[k])
                    dofy1[k] = d1.timetuple().tm_yday + d1.timetuple().tm_hour / 24. + d1.timetuple().tm_min / (24. * 60) + d1.timetuple().tm_sec / (24. * 3600)
                # end for
                if len(varnames) == len1:
                    lplt = ax[j].plot(dofy1, temp, linewidth = 1, label = varnames[il], color = colour_list[il])
                else:
                    lplt = ax[j].plot(dofy1, temp, linewidth = 1, color = colour_list[il])
                if limits1 is not None and len(limits1) == len1:
                    ax[j].set_ylim(limits1[j][0], limits1[j][1])

            # endif
            il += 1
            ax[j].locator_params(nbins=4, axis='y')

        # groups
        elif eval_group_order(2, group_first, len1, len2, len3, j, il, ig, groups):
            if revert != True:
                temp = [groups[ig][1:], groups[ig + 1][:]]
                dateTime = [dateTimes2[ig][1:], dateTimes2[ig + 1][:]]
            else:
                temp = [groups[ig][::-1], groups[ig + 1][::-1]]
                dateTime = [dateTimes2[ig][::-1], dateTimes2[ig + 1][::-1]]
            if yday == False:
                lplt1 = ax[j].plot(dateTime[0], temp[0], linewidth = 1, color = 'r', label = groupnames[ig])
                lplt2 = ax[j].plot(dateTime[1], temp[1], linewidth = 1, color = 'b', label = groupnames[ig + 1])
                pass
            else:
                dtime1 = dateTime[0]
                dtime2 = dateTime[1]
                dofy1 = numpy.zeros(len(dtime1))
                for k in range(0, len(dtime1)):
                    d1 = dates.num2date(dtime1[k])
                    dofy1[k] = d1.timetuple().tm_yday + d1.timetuple().tm_hour / 24. + d1.timetuple().tm_min / (24. * 60) + d1.timetuple().tm_sec / (24. * 3600)
                # end for
                dofy2 = numpy.zeros(len(dtime2))
                for k in range(0, len(dtime2)):
                    d2 = dates.num2date(dtime2[k])
                    dofy2[k] = d2.timetuple().tm_yday + d2.timetuple().tm_hour / 24. + d2.timetuple().tm_min / (24. * 60) + d2.timetuple().tm_sec / (24. * 3600)
                # end for
                d2 = dates.num2date(dtime2)
                lplt1 = ax[j].plot(dofy1, temp[0], linewidth = 1, color = 'r', label = groupnames[ig])
                lplt2 = ax[j].plot(dofy2, temp[1], linewidth = 1, color = 'b', label = groupnames[ig + 1])

            ig += 2
            ax[j].locator_params(nbins = 4, axis = 'y')
            ax[j].set_figwidth(ax[j].get_figwidth())-2

        # images
        #elif j > len1 + len2 - 1 and j < length and ii < len3:
        elif eval_group_order(3, group_first, len1, len2, len3, j, il, ig, groups):
            # Make some room for the colorbar
            fig.subplots_adjust(left = 0.07, right = 0.8)

            # Add the colorbar outside...
            box = ax[j].get_position()
            pad, width = 0.04, 0.016
            axb[ii] = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            #divider = make_axes_locatable(ax[j])
            #axb[ii] = divider.append_axes("right", size="3%", pad=0.1)

            if cblabel[ii] == None:
                cblabel[ii] = ""

            if sharey  == True and ii == int(len3 / 2 ) and len3 % 2 != 0:
                drawylabel = True
            elif sharey  == True:
                cblabel[ii] = ""

            display_img_temperatures_sub(fig, ax[j], axb[ii], dateTimes3[ii], imgs[ii], ticks[ii], maxdepths[ii],
                                         firstlogs[ii], maxtemps[ii], mindepths[ii], mintemps[ii],
                                         revert=revert, fontsize=fontsize, datetype=datetype, thermocline=False,
                                         interp=interp, ycustom=None, sharex=sharex,
                                         colorbar=True, cblabel=cblabel[ii])

            ii += 1
        # end if

        # LEGEND
        # blue_proxy = plt.Rectangle((0, 0), 1, 1, fc = "b")
        # ax[i].legend([blue_proxy], ['cars'])
        # ax[j].legend(shadow = True, fancybox = True)
        if label is not None:
            if len(varnames) == len1:
                handles, labels = ax[j].get_legend_handles_labels()
                if (j < len1 and il - 1 < len1) or (j > len1 - 1 and j < len1 + len2 and ig - 2 < len(groups)):
                    ax[j].legend(handles, labels)
            if label == 'frameless':
                plt.legend(frameon=False)

        # X-AXIS -Time
        # format the ticks
        if yday == False:
            formatter = dates.DateFormatter('%Y-%m-%d')
            # formatter = dates.DateFormatter('`%y')

            ax[j].xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
            ax[j].xaxis.set_minor_locator(mondays)
            fig.autofmt_xdate()

        if grid:
            # ax.xaxis.grid(True, 'major')
            ax[j].yaxis.grid(True, 'minor')
        else:
            ax[j].yaxis.grid(False),
        ax[j].xaxis.grid(True, 'major')


        if j < len1 and il - 1 < len1:
            ylabel = ylabels1[il - 1]
        # groups
        elif j > len1 - 1 and j < len1 + len2 and ig - 2 < len(groups):
            ylabel = ylabels2[ig - 2]
        elif j > len1 + len2 - 1 and j < length and ii - 1 < len3:
            ylabel = ylabels3[ii - 1]
        # end if

        if j == 2 and drawylabel == True:
            ax[j].set_ylabel(ylabel).set_fontsize(fontsize - 2)
        elif drawylabel == True:
            ax[j].set_ylabel(ylabel).set_fontsize(fontsize - 2)

        if  custom and title:
            title = ' Profile: %s' % custom[i]
            if drawylabel == True:
                ax[j].set_ylabel(custom[i])
            ax[j].set_title(title).set_fontsize(fontsize -1)
            
        if yday == False:
            if j > len1 - 1 and j < len1 + len2:
                ax[j].set_xlim(xmax = dateTime[0][len(dateTime) - 1])
                if j == length - 1:
                    ax[j].set_xlabel("Time").set_fontsize(fontsize)
            elif j < len1:
                ax[j].set_xlim(xmax = dateTime[len(dateTime) - 1])
                if j == length - 1:
                    ax[j].set_xlabel("Time").set_fontsize(fontsize-2)
        else:
            if j == length - 1:
                ax[j].set_xlabel("Day of Year").set_fontsize(fontsize-2)
                #ax[j].set_xlabel("Temperature $^\circ C$").set_fontsize(fontsize + 1)
            # end
            # ax[j].set_xlim(xmin = dofy1[0], xmax = dofy1[len(dofy1) - 1])

        # limits
        if limits != None:
            if limits[j] != None:
                ax[j].set_autoscale_on(False)
                ax[j].set_ylim(ymin = limits[j][0], ymax = limits[j][1])
                tk =  (limits[j][1] -  limits[j][0])/4.
                dy = tk/20.
                ax[j].set_yticks(numpy.arange(limits[j][0], limits[j][1]+dy, tk))
        else:
             # ax[j].legend(lplt, title = lg, shadow = True, fancybox = True)
            if tick != None:
               ax[j].set_yticks(tick[i][1])
               ax[j].set_yticklabels(tick[i][0])

        if maxdepth != None:
            if firstlog != None:
                mindepth = firstlog
            else:
                mindepth = 0
            ax[j].set_ylim(mindepth, maxdepth[i])

       

        i += 1

        if sharex and j < length - 1:
            plt.setp(ax[j].get_xticklabels(), visible = False, fontsize = fontsize - 3)
        elif sharex and j == length - 1:
            plt.setp(ax[j].get_xticklabels(), visible = True, fontsize = fontsize - 3)


        plt.setp(ax[j].get_yticklabels(), visible = True, fontsize = fontsize - 3)
    # end for
    # set labels visibility

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()

    # new seting to make room for labels
    if False: #this does not work.
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.9, wspace=0.02, hspace=None)

    plt.draw()
    plt.show()



def display_img_temperatures_sub(fig, ax, axb, dateTimes, temps, tick, maxdepth, firstlog, maxtemp, mindepth, mintemp,
                                 revert = False, fontsize = 20, datetype = 'date', thermocline = False, interp = None,
                                 ycustom = None, sharex=False, colorbar = False, cblabel = None):

    n = len(dateTimes[0]) - 1
    m = len(dateTimes)

    if False:  #numpy.ndim(temps) == 2:
        Temp = temps[:, 1:]
    else:
        Temp = numpy.zeros((m, n))
        i = 0
        for dateTime in dateTimes:
            j = 0
            c = temps[i]
            for t in c:
                # skip the first value because it is ZERO
                if j == 0:
                    j += 1
                    continue

                Temp[m - 1 - i, j] = t
                j += 1
                # print "%d , I=%d" % (j, i)
                if j > n - 2:
                    break
            # end for

            # Loop for shorter series (sensor malfunction) that need interpolation
            # Conditions:
            #    1- the  first and time series must be good
            #    2 - The bad time series can not be last or first (actually is implied from 1.
            # if j < n - 3 :
            #     prev = temps[i - 1]
            #     next = temps[i + 1]
            #     for jj in range(j, n - 1):
            #         # average the missing values
            #         Temp[m - 1 - i, jj] = (prev[jj] + next[jj]) / 2.0
            #         jj += 1
            #         # print "%d , I=%d" % (j, i)
            #     # end for
            #  # end  j < n - 3 :

            i += 1
            if i > m - 1 :
                break
        # end for dateTime in dateTimes:

    if datetype == 'dayofyear':
        dateTime = fft_utils.timestamp2doy(dateTimes[0][1:])
    else:
        dateTime = dateTimes[0][1:]


    #y = tick[1][::-1] #
    y = numpy.linspace(mindepth, maxdepth - firstlog, m)
    if interp != None:
        from scipy.interpolate import interp1d
        new_y = numpy.linspace(mindepth, maxdepth - firstlog, m * interp)
        #fint = interp1d(y, Temp.T, kind = 'quadratic')
        fint = interp1d(y, Temp.T, kind='slinear')
        newTemp = fint(new_y).T
        if revert == True:
            yrev = new_y[::-1]
        else:
            yrev = new_y

    else:
        if revert == True:
            yrev = y[::-1]
        else:
            yrev = y
        # if
        newTemp = Temp
    # end if interp

    X, Y = numpy.meshgrid(dateTime, yrev)
    if maxtemp != None and mintemp != None:
        im = ax.pcolormesh(X, Y, newTemp, shading = 'gouraud', vmin = mintemp, vmax = maxtemp, cmap = 'jet')#, norm = LogNorm())
    else:
        im = ax.pcolormesh(X, Y, newTemp, shading = 'gouraud', cmap = 'jet')#, norm = LogNorm())

    if colorbar:
        #cb = fig.colorbar(im, cax = axb, ax =ax)
        cb = fig.colorbar(im, cax=axb)
        cb.set_clim(mintemp, maxtemp)
        labels = cb.ax.get_yticklabels()
        for t in labels:
            t.set_fontsize(fontsize - 3)
        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=3)
        cb.locator = tick_locator
        cb.update_ticks()
        if cblabel:
            cb.set_label(cblabel)
            text = cb.ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size = fontsize-2)
            text.set_font_properties(font)
    # end if

    if ycustom != None:
        ylabel = ycustom
        plt.ylabel(ylabel).set_fontsize(fontsize - 2)
    # end if


    # reverse
    # ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_ylim(0, maxdepth)
    ax.set_yticks(tick[1])
    ax.set_yticklabels(tick[0])
    labels = ax.get_yticklabels()
    ax.tick_params(labelsize = fontsize - 1)

    # plt.setp(labels, rotation = 0, fontsize = fontsize)

    # format the ticks
    if sharex == False:
        if datetype == "date" :
            formatter = dates.DateFormatter('%Y-%m-%d')
            # formatter = dates.DateFormatter('`%y')
            # ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
            plt.setp(labels, rotation = 0, fontsize = fontsize)
            fig.autofmt_xdate()
        else:
            plt.xticks(fontsize = fontsize)
            # plt.xlabel("Day of year").set_fontsize(fontsize)

        ax.xaxis.set_minor_locator(hour)

        labels = ax.get_xticklabels()
    # end if sharex

    # draw the thermocline
    if thermocline:
        levels = [13]
        colors = ['w']
        linewidths = [1]
        ax.contour(X, Y, newTemp, levels, colors = colors, linewidths = linewidths, fontsize = fontsize)

# end display_img_temperatures_sub

def display_img_temperatures(dateTimes, temps, coeffs, k, tick, maxdepth, firstlog, maxtemp, revert = False,
                             fontsize = 20, datetype = 'date', thermocline = True, interp = None, ycustom = None,
                             cblabel = None, draw_hline = False, hline_freq = 2):
    '''
        Can interpolarte only if draw_hline == False

    '''

    if draw_hline and interp != None:
        print("Error: draw_hline and intepolation are not supportes simultaneously.")
        return

    n = len(dateTimes[0])
    m = len(dateTimes)
    if interp == None and draw_hline == True:
        Temp = numpy.zeros((m + 1, n - 1))
    else:
        Temp = numpy.zeros((m, n - 1))

    i = 0
    for dateTime in dateTimes:
        j = 0
        # c = coeffs[i]
        c = temps[i]
        for t in c:
            # skip the first value because it is ZERO
            if j == 0:
                j += 1
                continue

            Temp[m - 1 - i, j] = t
            j += 1
            # print "%d , I=%d" % (j, i)
            if j > n - 2:
                break
        # end for

        # Loop for shorter series (sensor malfunction) that need interpolation
        # Conditions:
        #    1- the  first and time series must be good
        #    2 - The bad time series can not be last or first (actually is implied from 1.
        if j < n - 3 :
            prev = temps[i - 1]
            next = temps[i + 1]
            for jj in range(j, n - 1):
                # average the missing values
                Temp[m - 1 - i, jj] = (prev[jj] + next[jj]) / 2.0
                jj += 1
                # print "%d , I=%d" % (j, i)
            # end for
         # end  j < n - 3 :

        i += 1
        if i > m - 1 :
            break

    # end for dateTime in dateTimes:


    fig = plt.figure()
    ax = fig.add_subplot(111)

    if datetype == 'dayofyear':
        dateTime = fft_utils.timestamp2doy(dateTimes[0][1:])
    else:
        dateTime = dateTimes[0][1:]

    if draw_hline == True:
        lsn = m + 1
    else:
        lsn = m

    y = numpy.linspace(0, maxdepth - firstlog, lsn)

    if interp != None and draw_hline == False :
        from scipy.interpolate import interp1d
        print("Interpolating on the vertical axis : n=%d" % interp)
        new_y = numpy.linspace(0, maxdepth - firstlog, m * interp)
        fint = interp1d(y[:], Temp.T, kind = 'cubic')
        newTemp = fint(new_y).T
        if revert == True:
            yrev = new_y[::-1]  # or use np.flipud()
        else:
            yrev = new_y
        X, Y = numpy.meshgrid(dateTime, yrev)
        im = ax.pcolormesh(X, Y, newTemp, shading = 'gouraud')  # , cmap = 'gray', norm = LogNorm())
    else:
        if revert == True:
            yrev = y[::-1]
        else:
            yrev = y
        X, Y = numpy.meshgrid(dateTime, yrev)
        im = ax.pcolormesh(X, Y, Temp)  # , cmap = 'gray', norm = LogNorm())

    if interp != None:
        divs = interp
    else :
        divs = 1

    # draw draw the line
    if draw_hline :
        for k in range(0, len(dateTimes) - 1):
            dt = dateTimes[k]
            if datetype == 'dayofyear':
                dt = fft_utils.timestamp2doy(dt[1:])
            yvalues = tick[1]

            if k != 0 and (k) % hline_freq == 0 and k + 1 < len(dateTimes):
                idx = k / hline_freq
                yh = (yvalues[idx - 1] + yvalues[idx]) / 2
                lis = [yh for ik in range(0, len(dt))]
                ax.plot(dt, lis, color = 'k', linewidth = '0.3')

    cb = fig.colorbar(im)
    cb.set_clim(0, maxtemp)
    labels = cb.ax.get_yticklabels()
    plt.setp(labels, rotation = 0, fontsize = fontsize - 2)
    if cblabel != None:
        cb.set_label(cblabel)
        text = cb.ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size = fontsize + 1)
        text.set_font_properties(font)

    if ycustom == None:
        ylabel = ' Depth [m]'
    else:
        ylabel = ycustom
    print("fontsize: %d" % fontsize)
    plt.ylabel(ylabel).set_fontsize(fontsize + 2)

    title = ' Temperature Profiles'
    ax.set_ylim(0, maxdepth)

    # THIS works only for the T-CHAIN April2012
    ax.set_yticks(tick[1])
    ax.set_yticklabels(tick[0])
    labels = ax.get_yticklabels()
    plt.setp(labels, rotation = 0, fontsize = fontsize - 2)

    # format the ticks
    if datetype == "date":
        formatter = dates.DateFormatter('%Y-%m-%d')
        # formatter = dates.DateFormatter('`%y')
        # ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(formatter)
        # ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        plt.setp(labels, rotation = 0, fontsize = fontsize)
        fig.autofmt_xdate()
    else:
        plt.xticks(fontsize = fontsize - 2)
        plt.xlabel("Day of the year").set_fontsize(fontsize + 2)

    ax.xaxis.set_minor_locator(hour)
    labels = ax.get_xticklabels()

    # draw the thermocline
    if thermocline:
        levels = [13]
        colors = ['k']
        linewidths = [0.6]
        ax.contour(X, Y, newTemp, levels, colors = colors, linewidths = linewidths, fontsize = fontsize)
    plt.show()


def display_vertical_velocity_profiles(coeffs, startdepth, revert = False, legendloc = 4, legend = None, \
                                       grid = False, xlabel = None, title = None, fontsize =20):

    vel = numpy.zeros(len(coeffs[0]))
    depth = numpy.linspace(startdepth, len(coeffs[0]) + startdepth, len(coeffs[0]))
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)

    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    lidx = 0
    for j in range(0,len(coeffs)) :
        for i in range(0, len(coeffs[j])):
            vel[i] = coeffs[j][i]
        if revert == True:
            reversed_temp = vel[::-1]
        else:
            reversed_temp = vel
        ax.plot(reversed_temp, depth, linestyle = ls[lidx], linewidth = 2.3)
        lidx += 1

    ax.grid(grid)

    if xlabel == None:
        xlabel = 'Velocity [$m s^{-1}$]'
    plt.xlabel(xlabel).set_fontsize(fontsize)
    ylabel = ' Depth [m]'
    plt.ylabel(ylabel).set_fontsize(fontsize)
    if title != None:
        plt.title(title).set_fontsize(fontsize)
    # reverse

    ax.set_ylim(ax.get_ylim()[::-1])  # [::1] reverses the array
    plt.xticks(fontsize = fontsize - 3)
    plt.yticks(fontsize = fontsize - 2)

    
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
    #-------------------------------------------------------------- center    10
    # prop={'size':14} font size  =14
    if legend != None:
        plt.legend(legend, loc = legendloc, prop = {'size':16})

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()
    plt.show()


def display_vertical_temperature_profiles(dateTimes, temps, coeffs, k, startdepth, profiles, revert = False, legendloc = 4):

    temp = numpy.zeros(len(dateTimes))
    depth = numpy.linspace(startdepth, len(dateTimes) + startdepth, len(dateTimes))
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)

    legend = []
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    lidx = 0
    for j in profiles :
        for i in range(0, len(dateTimes)):
            temp[i] = coeffs[i][j]
        if revert == True:
            reversed_temp = temp[::-1]
        else:
            reversed_temp = temp
        ax.plot(reversed_temp, depth, linestyle = ls[lidx], linewidth = 1.6)
        lidx += 1
        lg = '%s' % datetime.date.fromordinal(int(dateTimes[0][j]))
        legend.append(lg)

    ax.grid(True)

    xlabel = ' Temperature [$^\circ$C]'
    plt.xlabel(xlabel).set_fontsize(20)
    ylabel = ' Depth [m]'
    plt.ylabel(ylabel).set_fontsize(20)
    title = ' Temperature Profiles'
    # reverse

    ax.set_ylim(ax.get_ylim()[::-1])  # [::1] reverses the array


    plt.title(title).set_fontsize(22)
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
    #-------------------------------------------------------------- center    10
    # prop={'size':14} font size  =14
    plt.legend(legend, loc = legendloc, prop = {'size':12})

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()
    plt.show()

def display_simple_temperature_profiles(depths, temps, fname, revert = False, legendloc = 4, fontsize = 20):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)

    legend = []
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    lidx = 0
    for j in range(0, len(temps)):
        temp = temps[j]
        if revert == True:
            reversed_temp = temp[::-1]
        else:
            reversed_temp = temp
        ax.plot(reversed_temp, depths[j], linestyle = ls[lidx], linewidth = 1.6)
        lidx += 1
        
    ax.grid(True)

    xlabel = ' Temperature [$^\circ$C]'
    plt.xlabel(xlabel).set_fontsize(fontsize+1)
    ylabel = ' Depth [m]'
    plt.ylabel(ylabel).set_fontsize(fontsize+1)
    title = ' Temperature Profiles'
    # reverse

    ax.set_ylim(ax.get_ylim()[::-1])  # [::1] reverses the array

    #IF TITLE is needed 
    #plt.title(title).set_fontsize(22)
 
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
    #-------------------------------------------------------------- center    10
    # prop={'size':14} font size  =14
    fileName, fileExtension = os.path.splitext(fname)
    legend.append('%s' % fileName)
    leg = plt.legend(legend, loc = legendloc, prop = {'size':fontsize - 1})
    leg.get_frame().set_linewidth(0.0)
    
    plt.setp(ax.get_xticklabels(), visible = True, fontsize = fontsize -2)
    plt.setp(ax.get_yticklabels(), visible = True, fontsize = fontsize -2)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()
    plt.show()

def display_avg_vertical_temperature_profiles_err_bar_range(dateTimes, tempsarr, startdeptharr, revert = False,\
                                                            profiledates = None, doy = True,\
                                                            legendloc = 4, grid = False, title = None, sharex = True,\
                                                            fontsize = 20, xlabel =' Temperature [$^\circ$C]', depth_int =1, 
                                                            errbar=True, rangebar=True, debug = False):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    
    isVel = False
    if xlabel != None:
        try:
            slabel= xlabel.strip()
            if slabel[:3] == "Vel":
                isVel=True
        except:
            print("Parsing xlabel failed")

    format = 100 * len(dateTimes) + 10
    matplotlib.rcParams['legend.fancybox'] = True

    ax = numpy.zeros(len(dateTimes), dtype = matplotlib.axes.Subplot)

    i = 0
    for dateTime in dateTimes:
        temps = tempsarr[i]
        startdepth = startdeptharr[i]

        temp = numpy.zeros(len(dateTime))
        depth = numpy.linspace(startdepth, len(dateTime)*depth_int + startdepth, len(dateTime))


        if sharex and i == 0:
            ax[i] = fig.add_subplot(format + i + 1)
        else:
            ax[i] = fig.add_subplot(format + i + 1, sharex = ax[0])


        legend = []
        ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']

        lidx = 0
        avg_temp_arr = []
        avg_temp_arr_range_min = []
        avg_temp_arr_range_max = []
        avg_temp_arr_std = []

        for j in range(0, len(dateTime)):  # depth
            
            #if j != 18:
            temp_at_depth_j = temps[j]
            #else:
                #temp_at_depth_j = (temps[j - 1][1:] + temps[j + 1][1:]) / 2.
            
            if debug:
                fmt = "%y/%m/%d %H:%M"
                f = open("velocities-depths.txt", "a+")
                for t in range(0, len(temps[j])):
                    dt = matplotlib.dates.num2date(dateTime[j][t])
                    stringf= "depth: %d, date:%s vel:%f \n" % (j, dt.strftime(fmt), temp_at_depth_j[t])
                    f.write(stringf)  
                f.close()
            
                
            avg_temp = numpy.mean(temp_at_depth_j, axis = 0)
            avg_temp_std = numpy.std(temp_at_depth_j, axis = 0)
            avg_temp_min = numpy.min(temp_at_depth_j, axis = 0)
            avg_temp_max = numpy.max(temp_at_depth_j, axis = 0)

            avg_temp_arr.append(avg_temp)
            avg_temp_arr_std.append(avg_temp_std)
            
            if avg_temp < 0 and  avg_temp_min < 0:
                amin = abs(avg_temp_min-avg_temp)
            else :# avg_temp > 0 and  avg_temp_min < 0: 
                amin = abs(avg_temp-avg_temp_min) 
            
            if avg_temp < 0 and  avg_temp_max < 0:
                amax = abs(avg_temp-avg_temp_max)
            elif avg_temp < 0 and  avg_temp_max > 0:
                amax = abs(avg_temp_max-avg_temp) 
            else: 
                amax = abs(avg_temp_max-avg_temp)
            
            avg_temp_arr_range_min.append(amin)
            #avg_temp_arr_range_min.append(avg_temp - avg_temp_min)
            avg_temp_arr_range_max.append(amax)
            print("depth %d,  avg: %f max:%f min:%f  amax:%f amin:%f" % (j, avg_temp, avg_temp_max, avg_temp_min, amax, amin))

        if revert == True:
            reversed_temp = avg_temp_arr[::-1]
            reversed_min = avg_temp_arr_range_min[::-1]
            reversed_max = avg_temp_arr_range_max[::-1]
            reversed_std = avg_temp_arr_std[::-1]
        else:
            reversed_temp = avg_temp_arr
            reversed_min = avg_temp_arr_range_min
            reversed_max = avg_temp_arr_range_max
            reversed_std = avg_temp_arr_std

        if errbar:
            ax[i].errorbar(reversed_temp, depth, xerr = reversed_std, linewidth = 4.0, color = 'r',fmt='o', capsize = 6, capthick = 2)  # , fmt = 'o')
        if rangebar:
            ax[i].errorbar(reversed_temp, depth, xerr = [reversed_min, reversed_max], color = 'k', linewidth = 1.3, capsize = 5 ,capthick = 2)  # , fmt = 'd')
        ax[i].plot(reversed_temp, depth, linestyle = ls[lidx], linewidth = 5.2, color="black", label='_nolegend_')
        
        lidx += 1
        # lg = '%s' % datetime.date.fromordinal(int(dateTimes[0][j]))
        # legend.append(lg)

        if profiledates != None:
            legend = []
            snapshottemp = []
            dt = dateTime[0][2] - dateTime[0][1]
            for ii in range(0, len(profiledates)):
                snapshottemp.append([])
            
            if doy:
                occurence = []
                for ii in range(0, len(profiledates)):
                    dat = matplotlib.dates.num2date(profiledates[ii])
                    doy=str(dat.timetuple().tm_yday)
                    legend.append("DOY " +doy)
                    if type(dateTime[0]) == list:
                        datetim = numpy.array(dateTime[0])
                    else:
                        datetim = dateTime[0]
        
                    vel_day_of_profile =[]
                    
                    for j in range(0, len(dateTime)) :  # depth
                        kj=0
                        for dtm in dateTime[0]:  # depth
                            if dtm >= profiledates[ii] and dtm < profiledates[ii]+2:
                                vel_day_of_profile.append(tempsarr[i][j][kj])
                            kj+=1
                        vel_arr_day_of_profile =numpy.array(vel_day_of_profile)
                    
                        avg_vfile_temp = numpy.mean(vel_arr_day_of_profile, axis = 0)
                        snapshottemp[ii].append(avg_vfile_temp)
                        print("profile %d, depth %d avg t: %f" % (ii, j, avg_vfile_temp))
            else:
                # get indices
                occurence = []
                for k in profiledates:
                    if type(dateTime[0]) == list:
                        datetim = numpy.array(dateTime[0])
                    else:
                        datetim = dateTime[0]
                    occurence.append(numpy.where(abs(datetim - k) < dt/2.))
                   
               
                for ii in range(0, len(profiledates)):
                    dt = matplotlib.dates.num2date(profiledates[ii])
                    fmt = "%y/%m/%d %H:%M"
                    label = dt.strftime(fmt)
                    legend.append(label)
                    
                    for tm in temps:  # depth
                        snapshottemp[ii].append(tm[occurence[ii][0][0]])
                        print("time %f, date %s , ID=%d, temp %f" % (profiledates[ii], label, ii, tm[occurence[ii][0][0]]))
            #end else             
    
            for ii in range(0, len(profiledates)):
                #temp2 = temp2[::-1]
                if revert == True:
                    sn = snapshottemp[ii][::-1]
                else:
                    sn = snapshottemp[ii]
                    
                ax[i].plot(sn, depth, linestyle = ls[ii], linewidth = 2.6, label=legend[ii]) #, color = 'r')
            
            #ax[i].plot(temp2, depth, linestyle = ':', linewidth = 1.8, color = 'b')

        ax[i].grid(grid)
        ax[i].legend(loc = 2)
        xlabel = xlabel
        plt.xlabel(xlabel).set_fontsize(fontsize+2)
        ylabel = ' Depth [m]'
        plt.ylabel(ylabel).set_fontsize(fontsize+2)

        plt.setp(ax[i].get_xticklabels(), visible = True, fontsize = fontsize)
        plt.setp(ax[i].get_yticklabels(), visible = True, fontsize = fontsize)
        
        [ybot, ytop] = ax[i].get_ylim()[::-1]
        
        yres = (ybot - ytop)/50 
        ax[i].set_ylim(ybot+2*yres,ytop-2*yres )  # [::1] reverses the array

        if isVel:
            #print soem text
            [xleft, xright] = ax[i].get_xlim()
            
            xres = (xright - xleft)/50
            
            xx= xright-xres*8
            yy = ybot - yres*5 
            xxx= xleft+xres*3
            yyy = ybot - yres*5 
            
            print("xx = %f, yy = %f" % (xx,yy))
            ax[i].text(xx, yy, 'Lake', style='italic', fontsize=fontsize, bbox={'facecolor':'black', 'alpha':0.0, 'pad':10})
            ax[i].text(xxx, yyy, 'Embayment', style='italic', fontsize=fontsize, bbox={'facecolor':'black', 'alpha':0.0, 'pad':10})
            
        i += 1

    # end for

    if title != None:
        plt.title(title).set_fontsize(22)
    # reverse

    

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
    #-------------------------------------------------------------- center    10
    # prop={'size':14} font size  =14
    # plt.legend(legend, loc = legendloc, prop = {'size':12})

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    # fig.autofmt_xdate()
    plt.show()


def display_twinx(dates, data1, data2, label1, label2, ylabel1, ylabel2, doy=True, loc=0):
    from matplotlib import rc
    rc('mathtext', default='regular')
    if doy:
        dT = fft_utils.timestamp2doy(dates)
    else:
        dT= dates[:]


    fig = plt.figure()
    ax = fig.add_subplot(111)

    p1 = ax.plot(dT, data2, 'r-', label=label2) #value converted from cm to m
    ax.set_xlabel("Day of Year", fontsize=20)
    ax.set_ylabel(ylabel2, fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax2 = ax.twinx()
    p2 = ax2.plot(dT,  data1, 'c-', label = label1) #Converted value to profile average
    ax2.set_xlabel("Day of Year", fontsize=20)
    ax2.set_ylabel(ylabel1, fontsize=20)
    ax2.tick_params(axis='y', labelsize=16)

    #legend
    lns=p1 + p2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, frameon=False)

    ax.set_ylim([-0.4, 0.4])
    ax2.set_ylim([-0.15, 0.15])

    plt.show()
