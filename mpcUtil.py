import numpy as np
import math
import sys
MAX = sys.maxsize
T=math.pow(10, 15)

def mpcMulti(x1, y1, x2, y2) :

    u = np.random.randint(0, math.sqrt(MAX))

    v = np.random.randint(0, math.sqrt(MAX))

    w = u*v
    u1=np.random.randint(0, MAX)
    u2 = u - u1
    v1 =np.random.randint(0, MAX)
    v2 = v - v1
    w1 = np.random.randint(0, MAX)
    w2 = w - w1
    e = x1 - u1 + x2 - u2
    f = y1 - v1 + y2 - v2
    res1 = int((e * f + f * u1 + e * v1 + w1)/T)
    res2 = int((f * u2 + e * v2 + w2)/T)
    return res1, res2