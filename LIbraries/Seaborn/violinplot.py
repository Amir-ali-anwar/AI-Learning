# import seaborn as sns
# import matplotlib.pyplot as plt


# df= sns.load_dataset('titanic')

# print(df.head(5))
# sns.violinplot(data=df, x='sex', y='age',  hue='who', split=True)
# plt.title("Age Distribution by Gender (Titanic)")
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

df= sns.load_dataset('titanic')

df= df.dropna(subset=['age'])

print(df.head())

sns.set(style="whitegrid")

sns.violinplot(data=df, x='sex', y='age',  hue='survived', split=True)

plt.title("Age Distribution by Gender and Survival (Titanic)")
plt.xlabel("Sex")
plt.ylabel("Age")

plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()



