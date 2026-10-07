"""Program for demonstrating Python functionality."""

import matplotlib.pyplot as plt
from sklearn import linear_model

height = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
weight = [8, 10, 12, 14, 16, 18, 20]

plt.scatter(height, weight, color="black")
plt.xlabel("height")
plt.ylabel("weight")

bindushree = linear_model.LinearRegression()
bindushree.fit(height, weight)

x_height = [[12.0]]
print(bindushree.predict(x_height))

plt.show()