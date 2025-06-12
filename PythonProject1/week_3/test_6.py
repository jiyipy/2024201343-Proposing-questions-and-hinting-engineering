import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("../data/Advertising.csv")
print(df)

print(df.corr())
X = df[["TV"]]

Y = df[["sales"]]

# plt.scatter(X,Y)
# plt.show()

model = LinearRegression()
model.fit(X,Y)

y_pred = model.predict(X)
print(y_pred)
plt.scatter(X,Y)
plt.scatter(X,y_pred,c="red")
plt.show()
print(model.coef_)
print(model.intercept_)

mse = mean_squared_error(Y,y_pred)
print(mse)