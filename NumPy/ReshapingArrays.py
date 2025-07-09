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
# print(arr4[1,:2])


# 3-D Array

arr5 = np.array([ [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[13, 14, 15], [16, 17, 18], [19, 20, 21]]])

# print(arr5[0:5])
# print(arr5.shape)


# Reshape From 1-D to 3-D

arr6 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr6.reshape(2, 3, 2)

# print(newarr)


# Can We Reshape Into any Shape?
# Yes, as long as the elements required for reshaping are equal in both shapes.

# We can reshape an 8 elements 1D array into 4 elements in 2 rows 2D array but we cannot reshape it
# into a 3 elements 3 rows 2D array as that would require 3x3 = 9 elements.





arr7 = np.array([1,2,3,4,5,6,7])
# print(arr7[1:5:2])



# ==================================================================


arr8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
#   print(x)
  
  
  
# Iterating Array With Different Data Types

 arr9 = np.array([1, 2, 3])

for x in np.nditer(arr9, flags=['buffered'], op_dtypes=['S']):
# print(x)
 
 
 
#  Iterating With Different Step Size

  arr10 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr10[:, ::2]):
  print(x)