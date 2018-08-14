import numpy as np
a = np.zeros((2, 2))

print(a)

b = a.copy()
# np.insert(a, 1, np.array((1, 1)), 0)
a = np.vstack([np.array((1, 1)),a])
print(a)


#np.insert(b, 1, np.array((1, 1)), 1)
#a = np.hstack([b, np.array((1, 1))])


print(b)