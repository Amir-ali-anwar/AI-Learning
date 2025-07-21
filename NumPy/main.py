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




# =================================================================



arr12= np.array([[1,2,3,4],[5,6,7,8]])

print(arr12.shape)


# =================================================================


arr12 = np.array([1, 2, 3, 4], ndmin=5)

# print(arr12)
# print('shape of array :', arr12.shape)



# =================================================================


# Array Slicing & Boolean Indexing


arr13= np.array([1,2,3,4,5,6,7,8,9])

# print("arr[arr>2]",arr13[arr13>2])



# =================================================================


# Mathematical Functions


arr14 = np.array([1, 2, 3, 4, 5])

# print("Square Root:", np.sqrt(arr14))
# print("Exponential:", np.exp(arr14))
# print("Log (natural):", np.log(arr14))
# print("Sine:", np.sin(arr14))
# print("Cosine:", np.cos(arr14))

# =================================================================


# Aggregate/Statistical Functions

arr15 = np.array([1, 2, 3, 4, 5])

# print("Sum:", np.sum(arr))            # 15
# print("Mean:", np.mean(arr))          # 3.0
# print("Median:", np.median(arr))      # 3.0
# print("Standard Deviation:", np.std(arr))  # ~1.414
# print("Variance:", np.var(arr))   


# =================================================================


# Min/Max and Index Functions


arr15 = np.array([10, 20, 5, 40])

print("Min value:", np.min(arr15))          # 5
print("Max value:", np.max(arr15))          # 40
print("Index of Min:", np.argmin(arr15))    # 2
print("Index of Max:", np.argmax(arr15))    # 3



scores = np.array([50, 80, 45, 90, 60])
best_student_index = np.argmax(scores)
print("Top scorer is student at index:", best_student_index)



# =====================================================================

# Broadcasting 

A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([1, 0, -1])


print(A+B)



C = np.array([[1, 2, 3],
              [4, 5, 6]])

D = np.array([[10],
              [20]])

print(C + D)
