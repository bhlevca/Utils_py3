'''
Created on Jun 18, 2012

@author: bogdan
'''
import numpy as np
import scipy.signal  as sig
import matplotlib.pyplot as plt
import mlpy.wavelet as wave

x = np.array([1, 2, 3, 4, 3, 2, 1, 0])
wav = wave.dwt(x = x, wf = 'd', k = 6)

plt.plot(wav)
plt.show()
