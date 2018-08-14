import scipy as sp
import numpy as np
import math
import ufft.fft_utils as fft_utils
import matplotlib.pyplot as plt
from scipy import fftpack
import utools.phase_shift as phase_shift
import copy
from numpy import exp


def test_fft_ifft():
    
    # 1 Discrete-time domain representation
    A = 0.5            #amplitude of the cosine wave
    fc=10              #frequency of the cosine wave
    phase=30           #desired phase shift of the cosine in degrees
     
    fs=32*fc           #sampling frequency with oversampling factor 32
    
    t=np.linspace(0, 2-1/fs,int(2./(1./fs)))    #2 seconds duration
         
    phi = phase*np.pi/180 #convert phase shift in degrees in radians
    f1=fc
    f2=fc/4.
    x=A*np.cos(2*np.pi*f1*t+phi) + A/4.*np.cos(2*np.pi*f2*t+phi/2) #time domain signal with phase shift
   
    print ("f1:%f, f2=%f"%(f1,f2))
    fig=plt.figure()  
    plt.plot(t,x)     #plot the signal
    plt.show()
    
    # 2. Represent the signal in frequency domain
    N=int(2./(1./fs))           #FFT size
    X = 1/N*fftpack.fftshift(fftpack.fft(x,N)) #N-point complex DFT
    #X= fftpack.fft(x, N)  # DFT

    df=fs/N                                  #frequency resolution
    sampleIndex = np.linspace(-N/2,N/2-1,N)    #ordered index for FFT plot
    f=sampleIndex*df;                        #x-axis index converted to ordered frequencies
    print ("%d, %d" % (len(f), len(X)))
    #plt.stem(f,np.abs(X))                    #magnitudes vs frequencies
    plt.plot(f,np.abs(X))                    #magnitudes vs frequencies
    plt.xlabel('f (Hz)'); plt.ylabel('|X(k)|');
    plt.show()
    
    phase=np.arctan2(np.imag(X),np.real(X))*180/np.pi #phase information
    plt.plot(f,phase)                    #phase vs frequencies
    plt.show()
    
    X2=copy.copy(X) #store the FFT results in another array
    #detect noise (very small numbers (eps)) and ignore them
     
    threshold = max(np.abs(X))/10000               #tolerance threshold
    X2[abs(X)<threshold] = 0                       #maskout values that are below the threshold
    phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi          #phase information
    plt.plot(f,phase)                              #phase vs frequencies 
    plt.show()
    
    j=0
    for i in f:
        print ("i=%f, tres:%f" % (i, f2))
        if abs(i - f1) <= 1.02 or abs(i + f1) <= 1.02:
            print ("*** set freq %f to 0", f2)
            X[j]=0 
        j+=1 
    
    plt.plot(f,np.abs(X))                    #magnitudes vs frequencies
    plt.xlabel('f (Hz)'); plt.ylabel('|X(k)|');
    plt.show()
        
    # BH modify amplitude
    X3=copy.copy(X)
    
    #Reconstruct the signal
    x_recon = N*fftpack.ifft(fftpack.ifftshift(X3),N)             #reconstructed signal
    #x_recon= fftpack.ifft(X3)
    t = np.linspace(0, len(x_recon)-1, N)/fs #recompute time index
    #t=np.linspace(0, 2-1/fs,int(2./(1./fs)))    #2 seconds duration 
    plt.plot(t,x_recon)   
    plt.show()                           #reconstructed signal

def runningMean(x, N=100):
    '''
    N is the degree of smoothness - how many neighbouring points are taken in consideration
    '''
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def runningMeanFast(x, N=100):
    '''
    N is the degree of smoothness - how many neighbouring points are taken in consideration
    '''
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def ts_modify_runningMean(yd, Time, modifier, shift=0):
    rM = runningMean(yd, 40)
    df=yd-rM
    #tame by a factor of 5
    dfmod= df/modifier
    ydMod=rM+dfmod
    if shift != 0:
       ydMod = phase_shift.phase_shift(shift, ydMod)
    return ydMod 
    


def fft_modify(yd, Time, fftx, NFFT, mx, freq, power, modifiers):
    '''
    modifier is a list of dictionary entries  [ [{highcut:[val1,mod_fact1],  lowcut:val2}, {} ... ]   
    '''
    
    #adjust the frequencies that are needed
    #bp = np.zeros(len(fftx))
    ynew = copy.copy(fftx)
    amplitude = mx
    phase = np.angle(fftx)

    for i in range(len(freq)):
        for modf in modifiers:
            if freq[i] < modf['highcut'][0] and freq[i] >= modf['lowcut'][0]:
                amplnew = amplitude[i]/modf['highcut'][1]
                #H=1/ampl* math.exp(-ij* 0)
                print ("%d) bp=%f fftx=%f" %(i, ynew[i], fftx[i]))
                
                ynew[i]=amplnew*(math.cos(phase[i])+1j*math.sin(phase[i]))
                
                print ("%d) ratio:%f bp=%f fftx=%f" %(i, fftx[i]/ynew[i], ynew[i], fftx[i]))
        
    return ynew
    
def reconstruct(X,N):    
    x_recon = fftpack.ifft(X)
    #x_recon = N*fftpack.ifft(fftpack.ifftshift(X),N)             #reconstructed signal
    #return np.real(x_recon)
    return x_recon


def fft_analysis(Time, TimeSeries, draw = 'True', tunits = 'sec', log = 'linear', detrend =False):
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
            legend = ['Water levels']
            fft_utils.plotTimeSeries("Water levels", xlabel, ylabel, Time, TimeSeries, legend)
        # end

        # prepare for the amplitude spectrum analysis
        if tunits == 'day':
            factor = 86400
        elif tunits == 'hour':
            factor = 3600
        elif tunits == "min":
            factor = 60
        else:
            factor = 1

        dt_s = (Time[2] - Time[1]) * factor  # Sampling period [s]
        Fs = 1 / dt_s  # Samplig freq    [Hz]

        # nextpow2 = This function is useful for optimizing FFT operations, which are most efficient when sequence length is an exact power of two.
        #  does seem to affect the value of the amplitude
        # # NFFT = ufft.nextpow2(L)   # Next power of 2 from length of the original vector, transform length
        #
        NFFT = L
        # DETREND THE SIGNAL is necessary to put all signals oscillations around 0 ant the real level in spectral analysis
        if detrend:
            yd = sp.signal.detrend(TimeSeries)
        else:
            yd=np.array(TimeSeries)

        # Take ufft, padding with zeros so that length(fftx) is equal to nfft
        #w = np.hanning(100)
        fftx = fftpack.fft(yd, NFFT)  # DFT
        #fftx = fftpack.fft(yd, NFFT)  # DFT
        
        sFreq = np.sum(abs(fftx) ** 2) / NFFT
        sTime = np.sum(yd ** 2)

        # This is a sanity check
        #assert abs(sFreq - sTime) < eps

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
        #freq = np.array(list(range(0, NumUniquePts)))
        #freq = freq * Fs / NFFT  # 2
        # same as
        freq = np.fft.fftfreq(NFFT, d = dt_s) #[:NumUniquePts]

        return [yd, Time, fftx, NFFT, mx, freq, power]
    # end
    
    
def modify_series(time, series, modifier, tunit='min', shift = 0):
    [yd, Time, fftx, NFFT, mx, freq, power] = \
        fft_analysis(time, series, draw = False, tunits = tunit, log = 'linear', detrend = False)            
    ffty=fft_modify(yd, Time, fftx, NFFT, mx, freq, power, modifiers=modifier)

    modseries = reconstruct(ffty, NFFT)
    
    if shift != 0:
       modseries = phase_shift.phase_shift(shift, modseries)
    return [Time, modseries]
        
    
if __name__ == '__main__':
    # Test sinusoid
    # test_fft_ifft()
    # exit(1)
    
    test_output=True
    
    if test_output:
        path = '/home/bogdan/Documents/UofT/MITACS-TRCA/3DModel/Input_data/LO-water-levels-LocalTime'
        file1= "WL_DFO-13320-01-APR-2013.csv"
        file2= "WL_DFO-13320-01-APR-2013.csv_mod.tim"
        [Time1, TimeSeries1] = fft_utils.readFile(path, file1, date1st=True, sep=' ')
        [Time2, TimeSeries2] = fft_utils.readFile(path, file2, date1st=True, sep=' ')
        print ("s_time len:%d", len(Time1))
        print ("s_filtered len:%d", len(TimeSeries2)) 
        fig=plt.figure()
        plt.plot(Time1, TimeSeries1, 'b')
        plt.plot(Time2, TimeSeries2,'r')
        plt.legend(['Original Signal','Filtered Signal'])
        plt.xlabel('time [s]')
        plt.show()
        exit(1)
        
    path="/home/bogdan/Documents/UofT/PhD/Data_Files/2013/Hobo-Apr-Nov-2013/WL/csv_press_corr"
    
    filename="01-13-Aug-WL-Spadina_out.csv"
    [Time, TimeSeries] = fft_utils.readFile(path, filename, date1st=False)
    
    modifiers=[{'highcut':[2.777e-9,30],'lowcut':[2.777e-10,1]},    #0.00001 - 0.000001 cph
               {'highcut':[2.777e-8,30],'lowcut':[2.777e-9,1]},    #0.0001 - 0.00001 cph
               {'highcut':[2.777e-7,7],'lowcut':[2.777e-8,1]},    #0.001 - 0.0001 cph
               {'highcut':[2.777e-6,7],'lowcut':[2.777e-7,1]},    #0.01 - 0.001 cph
               {'highcut':[6.944e-5,7],'lowcut':[2.777e-6,1]},    #0.25 - 0.01 cph
               {'highcut':[0.000125,10],'lowcut':[6.944e-5,1]},    #0.45-0.25 cph
               {'highcut':[0.000611,10],'lowcut':[0.000125,1]},    #2.2-0.45 cph
               {'highcut':[0.0010277,10],'lowcut':[0.000611,1]},   #3.7-2,2 cph
               {'highcut':[0.0022777,10],'lowcut':[0.0010277,1]},  #8.2-3.7 cph
               {'highcut':[0.0010277,7],'lowcut':[1.1,1]},  
               {'highcut':[1.1,5],'lowcut':[1000.1,1]},  #
              ]

    detrend=False

    [yd, Time, fftx, NFFT, mx, freq, power, phase] = fft_analysis(Time, TimeSeries, draw = False, tunits = 'day', log = 'linear', detrend = detrend)
    
    ffty=fft_modify(yd, Time, fftx, NFFT, mx, freq, power, phase, modifiers)
    
    s_filtered = reconstruct(ffty, NFFT)
    
    if detrend:
        s_measured =  sp.signal.detrend(TimeSeries)
    else:
        s_measured =  TimeSeries
    s_time = Time
    fig=plt.figure()
    
       
    plt.plot(s_time, s_measured, 'b')
    plt.ylabel('Signal')
    plt.xlabel('time [s]')
     
    print ("s_time len:%d", len(s_time))
    print ("s_filtered len:%d", len(s_filtered)) 
    plt.plot(s_time,s_filtered,'r')
    plt.legend(['Original Signal','Filtered Signal'])
    plt.xlabel('time [s]')
    
    
    fig=plt.figure()
    ydMod=ts_modify_runningMean(yd, Time, 5)
    plt.plot(s_time, s_measured, 'b')
    plt.plot(s_time, ydMod,'r')
    plt.legend(['Original Signal','Filtered Signal'])
    plt.xlabel('time [s]')
    plt.show()
