#! /usr/bin/env python
"""
 CentralDiff.py is an example application of the central difference
 to numerically calculate the derivative of a function f(x) = x^3 sin(5x).

      Central Difference == Dc[f(x), h] = (f(x+0.5*h) - f(x-0.5*h)) / h

 The program compares analytical and numerical derivatives by graphing 
 results in the x range (1.0: 3.0).   The program then graphically explores 
 the values of the central difference as a function of h. 

 Paul Eugenio
 Florida State University
 Mar 24, 2015
"""



import numpy as np
import matplotlib.pyplot as plt


def F(x):
    """ f(x) = x^3 sin(5x) """
    return x**3 * np.sin(5*x)

def Dc(func, x, h=1e-8):
    """ 
     Central Difference == Dc[f(x), h] = (f(x+0.5*h) - f(x-0.5*h)) / h
    """
    return (func(x + 0.5*h) - func(x - 0.5*h)) / h


def dfdx(x):
    """ 
     analytic derivative 
        of f(x)  -> df/dx =  3x^3 sin(5x) + 5x^3 cos(5x)
    """
    return 3*x**2 * np.sin(5*x) + 5*x**3 * np.cos(5*x)

# Numpy Vectorization:
# vectorize user-defined functions so they
# can work with arrays of values producing an array of results  
vF = np.vectorize(F)
vD = np.vectorize(Dc)
vdFdx = np.vectorize(dfdx)


#
# main program
#
a,b = 1.0,3.0
N = 100

# generating points for plotting
x = np.linspace(a,b,N)
yF =   vF(x)
yD =  vD(vF,x)
ydF =  vdFdx(x)

# plotting F(x) & dFdx as a continuous line & plotting the
# central difference of F(x) as triangular points
fig1 = plt.figure(1)
plt.plot(x,yF, "-")
plt.plot(x,ydF, "-")
plt.plot(x,yD, "v")

plt.grid(True)
plt.title("Forward Difference")
plt.xlabel("x")
plt.legend([r"$f(x) = x^3 sin(5x)$", r"$\frac{df}{dx}$",r"$D^+ _h [f(x)]$"])
fig1.savefig("centralDiff-a.png")
fig1.show()


#
# Now fixing x=2 and calculating the central difference of F(x=2) as
# a function of h where Dc[F(x)] = (f(x+0.5h) - f(x-0.5h)) / h
#

x = 2
h = 1

hp,Dp = [],[]            # plotting points 

# calculate points h vs Dc[f(x),h]
for i in range(20):
    h *= 0.15
    Dp += [Dc(F, x, h)]
    hp += [h]

# generate figure
fig2 = plt.figure(2)
plt.semilogx(hp,Dp,"o-")
plt.xlabel("h")
plt.ylabel(r"$D^+ _h [f(x=2)]$")
fig2.savefig("centralDiff-b.png")
fig2.show()


input("Enter [return] to exit")
