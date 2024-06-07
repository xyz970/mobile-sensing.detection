import numpy as np
arr = np.array([0, 0.53], [1,   99.47])
max_arr = max(arr)
print(max_arr)
print(np.where(arr==max_arr)[0])
