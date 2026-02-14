# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.plot(x,y)

# plt.xlabel('Average pulse')
# plt.ylabel('Calorie Burnage')
# plt.show()



# ===============================================================


# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])


# plt.title('Sports Watch Data')
# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")

# plt.plot(x,y)
# # plt.grid(axis='x')
# plt.grid(axis='y')

# plt.show()





# ==================================================================





# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])


# plt.title('Sports Watch Data')
# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")

# plt.plot(x,y)
# # plt.grid(axis='x')
# plt.grid(color='green', linestyle='--', linewidth='1')

# plt.show()




# ==================================================================


# subplot()



import matplotlib.pyplot as plt
import numpy as np


x = np.array([0, 1, 2, 3])
y = np.array([6, 8, 1, 10])
z = np.array([6, 8, 1, 10])



# plt.subplot(2, 1, 2)
# plt.plot(x,y,z)


# plt.show()


# plt.subplot(2, 1, 2)
# plt.plot(x, y, label="jadu 1")
# plt.plot(x, z, label="Line 2")
# plt.legend()

# plt.show()






# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(6, 3, 1)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(2, 3, 2)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(2, 3, 3)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(2, 3, 4)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(2, 3, 5)
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(2, 3, 6)
# plt.plot(x,y)

# plt.show()




# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([0, 1, 2, 3])
# y1 = np.array([3, 8, 1, 10])
# y2 = np.array([1, 2, 5, 7])

# plt.subplot(2, 1, 1)
# plt.plot(x, y1)
# plt.title("Top Plot")

# plt.subplot(2, 1, 2)
# plt.plot(x, y2)
# plt.title("Bottom Plot")

# plt.tight_layout()
# plt.show()






# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([0, 1, 2, 3])
# y1 = np.array([3, 8, 1, 10])
# y2 = np.array([1, 2, 5, 6])

# fig = plt.figure()

# # Top subplot
# plt.subplot(2, 1, 1)
# plt.plot(x, y1)
# plt.title("Top Plot")

# # Bottom subplot
# plt.subplot(2, 1, 2)
# plt.plot(x, y2)
# plt.title("Bottom Plot")

# # Manually adjust layout
# fig.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1)

# plt.show()
