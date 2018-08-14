'''
Created on Jan 18, 2013

@author: bogdan
'''
from . import filters

class Filter(object):

    def __init__(self, doFiltering = None, lowcutoff = None, highcutoff = None, btype = 'band', order = 5):
        '''
        Constructor
        '''
        self.doFilter = doFiltering
        self.lowcutoff = lowcutoff
        self.highcutoff = highcutoff
        self.btype = btype
        self.order = order

    def butter_bandpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return filters.butter_bandpass(self.lowcutoff, self.highcutoff, fs, ord)

    def butter_highpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return  filters.butter_highpass(self.highcutoff, fs, ord)

    def butter_lowpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return filters.butter_lowpass(self.lowcutoff, fs, ord)

    def butterworth(self, data, fs):
        # returns [y, w, h, b, a]
        if self.doFilter:
            y, w, h, N = filters.butterworth(data, self.btype, self.lowcutoff, self.highcutoff, fs)
            return [y, w, h, N, None]
        else:
            return [data, None, None, None, None]
