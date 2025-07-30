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

print("\n📊 Visual Insights Report – Titanic EDA")

print("1. Female passengers had a much higher survival rate compared to males.")
print("2. Most passengers were in their 20s–30s.")
print("3. Survival was not strongly tied to age or fare alone.")
print("4. Female survivors were typically aged 20–40, while many males did not survive.")
print("5. Higher-class passengers (1st class) had better survival odds.")

plt.show()



