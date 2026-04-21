import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([500,800, 1000, 1200,1500]).reshape((-1, 1))
y = np.array([100,150,300,230,300])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Slope", model.coef_[0])

plt.scatter(X, y, color='black')
plt.plot(X, predictions, color='blue', linewidth=3)
plt.xlabel("Size of house")
plt.ylabel("Price of house")
plt.show()
