import numpy
import scipy.stats
import numpy.polynomial.polynomial as polynomial
import matplotlib.pyplot as plt

def interquartile_range(X, a = 25, b = 75, onlylower = False):
    """Calculates the Freedman-Diaconis bin size for
    a data set for use in making a histogram

    Arguments:
    X:  1D Data set
    a: lower quartile 0-> 100
    b: upperquartile  0 ->100

    b neesd to be greater thant a
    Returns:
    h:  F-D bin size
    """
    # check
    if b < a:
        raise Exception("upper quartile has to be greater than lower quartile")


    # First Calculate the interquartile range
    X = numpy.sort(X)
    upperQuartile = scipy.stats.scoreatpercentile(X, b)
    lowerQuartile = scipy.stats.scoreatpercentile(X, a)

    if onlylower:
        return lowerQuartile
    else:
        IQR = upperQuartile - lowerQuartile
        return IQR


# end interquartile_range


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return [r_value ** 2, slope, intercept, r_value, p_value, std_err]


def plot_regression(x, y, std, slope, intercept, point_labels = None, x_label = None, y_label = None, title = None,
                    r_value = None, p_value = None, fontsize = 20, bbox = False, show=False,
                    exclusions = None):

    print("intercept = %f, slope = %f" % (slope, intercept))
    fig, ax = plt.subplots()
    
    print ("x len:%d, std len: %d" % (len(x), len(std)))
    #    ax.errorbar(x, y, xerr = std, linewidth = 1.4, color = 'r',capsize = 6, capthick = 1.4, fmt='o')


    ax.plot(x, slope * x + intercept, 'b-', linewidth = 2.2)
   
    xmax = numpy.max(x)
    xmin = numpy.min(x)
    ymax = numpy.max(y)
    ymin = numpy.min(y)
    scalex = ax.xaxis.get_view_interval()
    xbins = len(ax.xaxis.get_gridlines()) - 1
    xn_per_bin = (scalex[1] - scalex[0]) / xbins

    scaley = ax.yaxis.get_view_interval()
    ybins = len(ax.yaxis.get_gridlines()) - 1
    ybin_sz = (scaley[1] - scaley[0])
    yn_per_bin = ybin_sz / ybins

    if point_labels != None:
        dx = scalex[0] / scalex[1] + 7. / xn_per_bin
        dy = scaley[0] / scaley[1] - 9. / yn_per_bin
        for i in range(0, len(point_labels)):
            print("label: %s" % point_labels[i])
            if  point_labels[i] in exclusions:
                col='red'
                if std != None:
                    ax.errorbar(x[i], y[i], xerr=std[i], linewidth=1.4, color=col, capsize=6, capthick=1.4, fmt='^')
                ax.plot(x[i], y[i], marker='^', color=col)
            elif point_labels[i] == 'Ah' or point_labels[i] == 'Al':
                col='green'
                if std != None:
                    ax.errorbar(x[i], y[i], xerr=std[i], linewidth=1.4, color=col, capsize=6, capthick=1.4, fmt='D')
                ax.plot(x[i], y[i], marker='D', color=col)
            else:
                col='black'
                if std != None:
                    ax.errorbar(x[i], y[i], xerr=std[i], linewidth=1.4, color=col, capsize=6, capthick=1.4, fmt='o')
                ax.plot(x[i], y[i], marker='o', color=col)


            ax.annotate(point_labels[i], xy = (x[i], y[i]), xycoords = 'data', xytext = (dx, dy) ,
                        textcoords = 'offset points', ha = 'left', va = 'top',
                        bbox = dict(fc = 'white', ec = 'none', alpha = 0.3) ,
                        color = col, fontsize = fontsize-7, zorder=0)

    if bbox:
        bbox_props = dict(boxstyle = "square,pad=0.3", fc = "white", ec = "b", lw = 1)
    else:
        bbox_props = None

    # transform = ax.transAxes ensures independence of text position from the plotting scale
    if r_value != None:
        text = "R$^2$=%4.2f" % r_value ** 2
        x0 = scalex[0] + (1.2) * xn_per_bin
        # x0 = scalex[0] + (xbins - 1.5) * xn_per_bin
        y0 = scaley[0] + (ybins - 0.8) * yn_per_bin
        # y0 = scaley[0] + (2) * yn_per_bin
        ax.text(0.02, 0.95, text, ha = 'left', va = 'center', bbox = bbox_props, fontsize = 14, transform = ax.transAxes)
    if p_value != None:
        text2 = "p-value=%2.5f" % p_value
        x0 = scalex[0] + (1.2) * xn_per_bin
        # x0 = scalex[0] + (xbins - 1.5) * xn_per_bin

        # y0 = scaley[0] + (1) * yn_per_bin
        ax.text(0.02, 0.9, text2, ha = 'left', va = 'center', bbox = bbox_props, fontsize = 14, transform = ax.transAxes)

    if x_label != None:
        plt.xlabel(x_label).set_fontsize(fontsize)
    if y_label != None:
        plt.ylabel(y_label).set_fontsize(fontsize)
    if title != None:
        plt.title(title).set_fontsize(fontsize + 2)

    plt.xticks(fontsize = fontsize - 1)
    plt.yticks(fontsize = fontsize - 1)

    xeps = abs((xmax - xmin) / 10.)
    yeps = abs((ymax - ymin) / 10.)

    ax.set_xlim(xmin = xmin - xeps , xmax = xmax + xmax / 8.)
    ax.set_ylim(ymin = ymin - yeps , ymax = ymax + ymax / 8.)

    plt.grid(False)
    if show:
        plt.show()
    
def residuals(x, y, deg = 1):    
    p_coef, residuals, rank, singular_values, rcond = numpy.polyfit(x, y, deg, rcond=None, full=True, w=None, cov=False)
    return residuals


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6]
    y = [6, 5, 4, 3, 2, 1]
    x = [5.05, 6.75, 3.21, 2.66]
    y = [1.65, 26.5, -5.93, 7.96]
    print(rsquared(x, y))
