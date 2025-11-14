
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([3,4,2,5,6])

model = LinearRegression()
model.fit(X,y)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Model")

plt.savefig("regression_graph.png")
plt.show()
