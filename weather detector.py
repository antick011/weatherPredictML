import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('weatherHistory.csv')
#print(data.head())
#print(data.columns)
data = data.dropna()
le = LabelEncoder()
data['Precip Type'] = le.fit_transform(data['Precip Type'])
X = data[['Temperature (C)', 'Humidity']]
y = data['Apparent Temperature (C)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
mse = mean_squared_error(y_test, prediction)
#print(f"Mean Squared Error: {mse}")
plt.plot(y_test.values[:100], label="Actual")
plt.plot(prediction[:100], label="Prediction")
plt.legend()
plt.show()