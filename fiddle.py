import numpy as np

threshold = 0.1

vec = np.array([0.3000000, 0.7, 0.0, 0.0, 0.0])
vec2 = np.array([0.3, 0.7, 0.0, 0.0, 0.0])

ret = np.array_equal(vec, vec2)

print(ret)