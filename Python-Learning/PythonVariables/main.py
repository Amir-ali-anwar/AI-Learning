
# Many Values to Multiple Variables
x,y,z= "orange", "yellow","pink"

print(x)
print(y)
print(z)

#  One Value to Multiple Variables
x,y,z= "Amir", "Amir","Amir"

print(x)
print(y)
print(z)

# Unpack a Collection
fruits= ['apple', 'banana','mango']
x,y,z= fruits
print(y)
print(z)
# Output Variables



x = "python is awesome"

print(x)



x = "Python"
y = "is"
z = "awesome"
print(x, y, z)
print(x + y + z)



# Global Variables


q = "awesome"

def myfunc():
  print("Amir is " + q)

myfunc()

# The global Keyword

def myFun():
    global customVar
    customVar = 'yahoo'
    return customVar

print(myFun())

x = "awesome"

def myfunc():
  global x
  x = "fantastic"

myfunc()

print("Python is " + x)


print(myfunc())