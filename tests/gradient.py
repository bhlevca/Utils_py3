import numpy
import scipy

x = [ 0,  0.1,  0.2,  0.3, 0.4]
y = [ 0.0000,  0.0819,  0.1341,  0.1646,  0.1797]
dx =0.1

#y = [ 0.85866,  0.86289,  0.86710,  0.87129]
#dx =0.01

#y = [ 1.0628, 1.3961, 1.5432,  1.7349, 1.8423, 2.0397]
#dx = 0.4

grad = numpy.gradient(y,0.1, edge_order=1)

print(grad)


grad = numpy.gradient(y,0.1, edge_order=2)

print(grad)

dy = numpy.diff(y)/dx

print(dy)

def eddy_viscosity(V, W, Z, dz):
    '''
    Using the method from: 
                W. J. Shaw and J. H. Trowbridge 
                The Direct Estimation of Near-Bottom Turbulent Fluxes in the Presence of Energetic Wave Motions,
                J. Atmos. Oceanic Technol., 2001
                
                numpy.cov(a, b) retuns
                    |cov(a,a)  cov(a,b)|
                    |cov(a,b)  cov(b,b)|
    '''
    
    covariance = numpy.cov(V,W)[0][1]
    grad       = numpy.gradient(V,dz)
    Kv = covariance/grad.mean()
    return Kv 


print("Kv=",eddy_viscosity(y,x, 1, 0.1))