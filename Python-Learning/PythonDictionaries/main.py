thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}


print(thisdict['brand'])


# Duplicates Not Allowed

thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964,
  "year": 2025
}
print(thisdict['year'])

x = thisdict.keys()

print(x)

values = thisdict.values()

print(values)


items = thisdict.items()

print(items)