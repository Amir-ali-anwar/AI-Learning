thislist= ["apple", "banana", "cherry"]

print(thislist)
print(type(thislist))

thislist.append("grapefruit")
thislist.insert(1,"Mango")
print(thislist)

thisTuple= ('Kiwi', 'Orange')

thislist.extend(thisTuple)

print(thislist)

# Removing item of list

thislist.remove('apple')

print(thislist)

# Iterating the list
newlist=[]
for x in thislist:
    if "o" in x:
        newlist.append(x)
  
  
print('new list',newlist)
        
        

    