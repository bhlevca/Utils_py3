import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import numpy as np

def plot_n_Array_with_CI(title, xlabel, ylabel, x_arr, y_arr, ci05, ci95, legend = None, linewidth = 0.8, ymax_lim = None, log = 'linear', \
                         fontsize = 20, plottitle = False, grid = False, twoaxes = False, ylabel2 = None, ymax_lim2 = None, drawslope = False,\
                          threeaxes=False, ylabel3 = None, ymax_lim3 = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')


    ax=[]
    Y_labels=[]
    Y_labels.append(ylabel)
    if  twoaxes or threeaxes:
        ax.append(host_subplot(111, axes_class=AA.Axes))
        plt.subplots_adjust(right=0.75)
    else:
        ax.append(fig.add_subplot(111))
    if twoaxes or threeaxes and len(y_arr) >1:
        ax.append(ax[0].twinx())
        Y_labels.append(ylabel2)
    if threeaxes and len(y_arr) >2:
        ax.append(ax[0].twinx())
        offset = 70
        new_fixed_axis = ax[2].get_grid_helper().new_fixed_axis
        ax[2].axis["right"] = new_fixed_axis(loc="right", axes=ax[2],offset=(offset, 0))
        ax[2].axis["right"].toggle(all=True)   
        Y_labels.append(ylabel3)
    #endif
    for i in range(0,len(x_arr)):
            
        if log == 'loglog':
                #line = axs.loglog(x, y, linestyle = lst[i], linewidth = lwt, basex = 10, color = colors[i], label = legend[i])
                #axs.set_yscale('log')
                ax[i].set_xscale('log')
                ax[i].set_yscale('log')
                # axs.semilogx(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i])
                print("xscale=log; yscale=log")
        elif log == 'log':
            ax[i].set_xscale('log')
            ax[i].set_yscale('linear')
            #line = axs.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i], label = legend[i])
            print("xscale=linear; yscale=log")
        else:
            #formatter =  EmbeddedSciFormatter(10, 1)
            #axs.yaxis.set_major_formatter(formatter)
            #line = axs.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i], label = legend[i])
            print("xscale=linear; yscale=linear")


    lines = []
    Xmax = []
    Ymax = []
    Xmin = []
    Ymin = []
 
    lst = ['-', '--', '-.', ':', '-', '--', ':', '-.']
    lst = ['-', '-', '-', '-', '-', '-', '-', '-']
    colors = ['b', 'c', 'r', 'k', 'y', 'm', 'aqua', 'k']
    
   
    i = 0
    for a in x_arr:
        x = x_arr[i][3:]
        y = y_arr[i][3:]
        
        
        if len(x_arr) < 5:
            lwt = 2.6 - i * 0.2
        else:
            lwt = 1 + i * 0.6

        if threeaxes and i == 2:
            axs = ax[2] 
        if (threeaxes and i == 1) or  (twoaxes and i == 1):
            axs = ax[1]
        else:
            axs = ax[0]
        
       
        line = axs.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i], label = legend[i])
        
        if drawslope:
            # select only the data we need to asses
            x1 = 0.1; x2 = 0.8
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
            axs.plot(xs, yfit, color = 'r')
            
        lines += line
        i += 1
    # end for

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
            axs.fill_between(x, y1, y2, where = y2 > y1, facecolor = [sd, sd, sd], alpha = 0.5, interpolate = True, linewidth = 0.001)
            # axs.fill_between(x, y1, y2, facecolor = 'blue', alpha = 0.5, interpolate = True)
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
        # ax.xaxis.grid(True, 'major')
        vertices = [(Xmin[i], Ymin[i]), (Xmax[i], Ymax[i])]
        
        ax[i].xaxis.grid(grid, 'minor')
        ax[i].yaxis.grid(grid, 'minor')
        plt.setp(ax[i].get_xticklabels(), visible = True, fontsize = fontsize)
        plt.setp(ax[i].get_yticklabels(), visible = True, fontsize = fontsize)
        #plt.setp(ax[i].get_xlabels(), visible = True, fontsize = fontsize+1)
        #plt.setp(ax[i].get_ylabels(), visible = True, fontsize = fontsize+1)
        
        ax[i].grid(grid)
        ax[i].set_ylabel(Y_labels[i], fontsize = fontsize)
        ax[i].set_xlabel(xlabel, fontsize = fontsize)
        #fontd = {'family' : 'serif',
        #         'color'  : 'darkred',
        #         'weight' : 'normal',
        #         'size'   : 'large',
        #}
        #ax[i].yaxis.set_label_text(ylabel, fontdict=fontd)
        if i  == 0 :  side = 'left' 
        else: side = "right"       
        
        if  twoaxes or threeaxes:
            ax[i].axis[side].label.set_fontsize(fontsize)
            ax[i].axis["bottom"].label.set_fontsize(fontsize)
        

    plt.xlim(xmin=np.min(Xmin), xmax=np.max(Xmax))
    #this causes some problems
    #plt.ylim(ymin=np.min(Ymin), ymax=np.max(Ymax))

    if plottitle:
            plt.title(title, fontsize = fontsize)

    if legend is not None:
        labs = [l.get_label() for l in lines]
        #legnd = ax[0].legend(lines, labs, loc = 'upper right')
        legnd = ax[0].legend(lines, labs, loc = 3)
        for label in legnd.get_texts():
            label.set_fontsize(fontsize - 6)

    plt.show()
# end

fs = 100
T = 1.0
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint = False)
                   
sin1 = 5 + np.sin(2 * np.pi * t * 10) * 4.0  # (B) sin1
c11 = 0.9 * sin1  
c12 = 1.1 * sin1 

sin2 = 5 + np.sin(2 * np.pi * t * 20) * 2.0  # (B) sin2
c21 = sin2 - sin2/5. 
c22 = sin2  + sin2/5. 

sin3 = 12 + np.sin(2 * np.pi * t * 3) * 6.0  # (B) sin3
c31 = sin3 - sin3/10. 
c32 = sin3  + sin3/10. 

xa = [t,t,t]
#xa = [t]
ya= [sin1, sin2, sin3]
#ya= [sin1]
ci05 =  [c11, c21, c31]
ci95 =  [c12, c22, c32]
#ci05 =  [c11]
#ci95 =  [c12]

title = "Test"
xlabel = 'Time'
ylabel1 = "Graph1"
ylabel2 = "Graph2"
ylabel3 = "Graph3"

plot_n_Array_with_CI(title, xlabel, ylabel1, xa, ya, ci05, ci95, legend = ["Graph1", "Graph2", "Graph3"], \
                                                    log = "loglog", fontsize = 20, plottitle = None, grid = True, ymax_lim = None,\
                                                    twoaxes = False, ylabel2 = ylabel2, ymax_lim2 = None, drawslope = False,\
                                                    threeaxes = True, ylabel3 = ylabel3, ymax_lim3 = None)