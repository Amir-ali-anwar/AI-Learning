import seaborn as sns
import matplotlib.pyplot as plt

iris= sns.load_dataset('iris')

print(iris)

print(iris['species'].unique())

sns.pairplot(iris, hue='species')

plt.suptitle("Iris Pairplot", y=1.02)

plt.show()