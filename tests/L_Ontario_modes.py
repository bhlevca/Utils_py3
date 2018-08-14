import math

# Lake ontario modes
L = 300000
H = 86
g = 9.81

modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 16]

for n in modes:
    T = 2 * L / n / math.sqrt(g * H) / 3600
    print("T[%d]=%f [hours]" % (n, T))
