import numpy as np
import matplotlib.pyplot as plt
soa = np.array([ [0, 0, 3, 2], [0, 0, 1, 1], [0, 0, 9, 9]])
X, Y, U, V = list(zip(*soa))
plt.figure()
ax = plt.gca()
# ax.quiver([0], [0], [1], [1], uscale = 1, angles = 'xy', scale_units = 'xy')
# ax.quiver([0, 0], [0, 0], [5, 5], [5, 0], angles = 'xy', scale_units = 'xy', scale = 1)

arr = plt.Arrow(0, 0, 5, 5)
plt.gca().add_patch(arr)
# ax.quiver(X, Y, U, V, angles = 'xy', scale_units = 'xy', scale = 1)
ax.set_xlim([-1, 10])
ax.set_ylim([-1, 10])
plt.draw()
plt.show()
