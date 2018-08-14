import pylab as plt
import numpy as np
import scipy
from scipy.signal import butter, lfilter, filtfilt
numsamples=1000
f0=1/3600
x=np.linspace(0,36000,numsamples)
y=np.sin(2*np.pi*f0*x)+np.sin(2*np.pi*f0/2*x)+0.5*np.sin(2*np.pi*f0/3*x)+0.25*np.sin(2*np.pi*f0/51*x)
fs=numsamples/(max(x)-min(x))
nyquist=0.5*fs
#fstart=(3/4)*f0/nyquist
#fstop=(4/3)*f0/nyquist
#a,b  = butter(2,[fstart,fstop],'bandstop', analog=False)
fstart=(0.5)*f0
fstop=(156)*f0
a,b  = butter(9,[fstart,fstop],'bandstop', analog=True)
fy = filtfilt(a,b,y) #, axis=-1, zi=None)
plt.plot(y)
plt.plot(fy)
plt.show()