import math
import pylab

pylab.plot(list(range(-50,50)), list(range(-50,50)))

opt = {'head_width': 0.4, 'head_length': 0.4, 'width': 0.2,
        'length_includes_head': True}
for i in range(1, 360, 20):
    x = math.radians(i)*math.cos(math.radians(i))
    y = math.radians(i)*math.sin(math.radians(i))

    # Here is your method.    
    arr = pylab.Arrow(0, 0, x, y, fc='r', alpha=0.3)
    pylab.gca().add_patch(arr)

    # Here is the proposed method.
    pylab.arrow(0, 0, x, y, alpha=0.8, **opt)

pylab.show()
