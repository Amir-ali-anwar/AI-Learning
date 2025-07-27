# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.displot([0, 1, 2, 3, 4, 5], kind='kde')

# plt.show()


# import seaborn as sns

# import matplotlib.pyplot as plt

# import pandas as pd

# sns.set_theme(style='whitegrid')

# df = sns.load_dataset('titanic')

# print(df.head())
# print(df.info())
# print(df.describe())

# # sns.countplot(x='sex', hue='survived', data=df)

# # plt.title("Passenger Gender Count")
# # plt.show()

# # sns.histplot(data=df, x='age')
# # plt.title("Distribution of Passenger Ages")
# # plt.show()


# sns.scatterplot(data=df, x='age', y='fare', hue='sex', style='survived', size='fare')
# plt.title("Age vs Fare (style by survival, size by fare)")
# plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set theme
sns.set_theme(style='whitegrid')

# Load dataset
df = sns.load_dataset('titanic')

# Explore data
print(df.head())
print(df.info())
print(df.describe())

# Scatter plot
sns.scatterplot(data=df, x='age', y='fare', hue='sex', style='survived', size='fare')
plt.title("Age vs Fare (style by survival, size by fare)")
plt.show()

# Correlation heatmap
# Only select numeric columns for correlation
corr = df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(10, 6))  # Optional: make the figure wider
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
