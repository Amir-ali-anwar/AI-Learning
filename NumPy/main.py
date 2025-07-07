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

print("arr.dtype",arr.dtype)

print('4th element on 2nd row: ', arr6[1, 3])  # Output: 9


arr7 = np.array([1, 2, 3, 4], dtype='S')

print(arr7)
print(arr7.dtype)



# =================================================================



arr7= np.array([1,2,3,4], dtype='i4')

print("arr7",arr7)
print("arr7.dtype",arr7.dtype)


# =================================================================



arr8 = np.array(['1', '2', '3'], dtype='i')

print("arr8",arr8)


# =================================================================


arr9 = np.array([1.1, 2.1, 3.1])

newarr1=  arr9.astype('i')
print(newarr1)
print(newarr1.dtype)



# =================================================================

# COPY:
arr10 = np.array([1, 2, 3, 4, 5])
x = arr10.copy()
arr10[0] = 42

print(arr10)
print(x)


# =================================================================


arr11 = np.array([2, 3, 4, 5])
viewArray = arr11.view()
arr11[0] = 42

print("arr11",arr11)
print("viewArray",viewArray)



# =================================================================



# Check if Array Owns its Data



xCopyArray = arr11.copy()
yViewArray = arr11.view()

print(xCopyArray.base)
print(yViewArray.base)





