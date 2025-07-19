import numpy as np


# print(np.random.randint(1, 10))            
# print(np.random.randint(1, 10, size=5))            

# print(np.random.randint(1, 50, (3,4)))            


arr= [10,20,30,40,50]

# print(np.random.choice(arr))


arr= np.array([1,2,3,4,5])

np.random.shuffle(arr)
# print(arr)


np.random.seed(42)  # Fix the randomness
print(np.random.randint(1, 10, size=5))
