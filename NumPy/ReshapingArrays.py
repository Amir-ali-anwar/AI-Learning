import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

# print(newarr)

# =============================================


arr1 = np.array([[1,2,3,4,5,6], [1,2,3,4,5,6]])

# print(arr1.ndim)

arr2 = np.array([[[1,2,3,3,4,5]]])

# print(arr2.ndim)

arr3 = np.array([1,2,3,4,5], ndmin=5)

# print(arr3)


# Accessing Array
arr4 = np.array([[1,2,3],[4,5,6]])
print(arr4[1,:2])



