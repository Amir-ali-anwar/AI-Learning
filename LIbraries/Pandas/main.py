import pandas as pd

mydataSet = {
    'cars':['BMW', "Volvo","Ford"],
    "Passings":[3,4,5]
}

myVar =  pd.DataFrame(mydataSet)

# print(myVar)

# ==========================================


a = [1, 7, 2]

myVar1= pd.Series(a)
print("myVar1",myVar1)

#  Create Labels

myVar2= pd.Series(a, ['x',"y",'z'])

# print("myVar2",myVar2)


# ============================================

# Locate Row


print(myVar.loc[0])

