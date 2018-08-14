import numpy
def interpolateData(interval, data, dateTime):
    '''
    Interpolate the data to "interval"
    '''
    ratio = (dateTime[2]-dateTime[1]) / interval
    # interpolated values
    dataInterp = numpy.interp(numpy.linspace(dateTime[0], dateTime[len(dateTime)-1], int(ratio * len(data))), \
                              dateTime, data)
    dateTimeInterp = numpy.interp(numpy.linspace(dateTime[0], dateTime[len(dateTime)-1], int(ratio * len(data))), \
                                  dateTime, dateTime)
    
    return [dataInterp, dateTimeInterp]