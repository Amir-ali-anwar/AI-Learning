# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])



# plt.scatter(x,y)

# plt.show()


# import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5, 6]
# y = [2, 4, 5, 4, 5, 7]

# plt.scatter(x, y)
# plt.title("Check Relationship")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()




import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [2, 4, 5, 4, 6, 7]

plt.scatter(x, y)
plt.title("Scatter Plot - Relationship Between X and Y")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.show()
