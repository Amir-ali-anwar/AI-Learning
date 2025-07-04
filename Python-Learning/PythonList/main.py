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