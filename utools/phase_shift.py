import numpy

def phase_shift(key, array):
    if type(array) == numpy.ndarray:
        return numpy.roll(array,key)
    else: 
        return array[-key:] + array[:-key]



if __name__ == '__main__':
    
    a=[1,2,3,4,5,6,7,8,9,0]
    print ("initial:", a)
    #test  positive
    print ("positive:", phase_shift(3,a) )
    #test  negative
    print ("negative:", phase_shift(-3,a) )
    
    a=numpy.array([1,2,3,4,5,6,7,8,9,0])
    print ("initial ndarray:", a)
    #test  positive
    print ("positive:", phase_shift(3,a) )
    #test  negative
    print ("negative:", phase_shift(-3,a) )
