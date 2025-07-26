# import matplotlib
import matplotlib.pyplot as plt

import numpy as np


# print(np.__version__)

# Xpoints= np.array([0,6])
# Ypoints= np.array([0,250])

# plt.plot(Xpoints,Ypoints) 

# plt.show() 



# ============================



# Xpoints= np.array([0,6])
# Ypoints= np.array([0,250])

# plt.plot(Xpoints,Ypoints, 'o') 

# plt.show()


# ===============================


# Multiple Points


# import matplotlib.pyplot as plt


# xpoints = np.array([1, 2, 6, 8])
# ypoints = np.array([3, 8, 1, 10])

# plt.plot(xpoints, ypoints)
# plt.show()



# ===============================


# Default X-Points


ypoints = np.array([3, 8, 1, 10, 5, 7])

# plt.plot(ypoints, marker='o')
# plt.plot(ypoints, marker='*')
plt.plot(ypoints, marker= "o", ms=10 , mec="r", mfc="g")

plt.savefig('saveplot.jpg')
plt.show()