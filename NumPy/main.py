import numpy as np

# print(np.__version__)


arr = np.array([1,2,3,4,5])
# arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type (arr))


# 1-D Array

arr1= np.array((1))

print('arr1',arr1)




d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print('d.ndim',d.ndim)


#  Higher Dimensional Arrays


arr4 = np.array([1, 2, 3, 4], ndmin=5)

print("arr4",arr4)


# 

arr5 = np.array([1, 2, 3, 8])

print(arr5[2] + arr5[3])



arr6 = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('4th element on 2nd row: ', arr6[1, 3])  # Output: 9