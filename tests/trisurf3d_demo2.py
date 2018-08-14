import numpy as np
y,x = np.mgrid[1:10:10j, 1:10:10j] # returns 2D arrays
# You have 1D arrays that would make a rectangular grid if properly reshaped.
y,x = y.ravel(), x.ravel()  # so let's convert to 1D arrays
z = x*(x-y)
colors = np.cos(x**2) - np.sin(y)**2

from scipy.interpolate import RectBivariateSpline
# from scipy.interpolate import interp2d # could 've used this too, but docs suggest the faster RectBivariateSpline

# Define the points at the centers of the faces:
y_coords, x_coords = np.unique(y), np.unique(x)
y_centers, x_centers = [ arr[:-1] + np.diff(arr)/2 for arr in (y_coords, x_coords)]

# Convert back to a 2D grid, required for plot_surface:
Y = y.reshape(y_coords.size, -1)
X = x.reshape(-1, x_coords.size)
Z = z.reshape(X.shape)
C = colors.reshape(X.shape)
#Normalize the colors to fit in the range 0-1, ready for using in the colormap:
C -= C.min()
C /= C.max()

interp_func = RectBivariateSpline(x_coords, y_coords, C.T, kx=1, ky=1) # the kx, ky define the order of interpolation. Keep it simple, use linear interpolation.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = ax.plot_surface(X,Y,Z,
    facecolors=cm.hot(interp_func(x_centers, y_centers).T),
    rstride=1,  cstride=1) # only added because of this very limited dataset
plt.show()