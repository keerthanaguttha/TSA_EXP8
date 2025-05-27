# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

```
Devloped by: Guttha Keerthana
Register Number: 212223240045
Date: 23-05-2025
```


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```
Read the dataset
```
data = pd.read_csv('smooth_data.csv')
```
Focus on the 'yahoo_price' column
```
yahoo_data = data[['yahoo_price']]
```
Display the shape and the first 10 rows of the dataset
```
print("Shape of the dataset:", yahoo_data.shape)
print("First 10 rows of the dataset:")
print(yahoo_data.head(10))

Plot Original Dataset 
```
plt.figure(figsize=(12, 6))
plt.plot(yahoo_data['yahoo_price'], label='Original Yahoo Price')
plt.title('Original Yahoo Price Data')
plt.xlabel('Date')
plt.ylabel('Yahoo Price')
plt.legend()
plt.grid()
plt.show()
```
Moving Average
Perform rolling average transformation with a window size of 5 and 10
```
rolling_mean_5 = yahoo_data['yahoo_price'].rolling(window=5).mean()
rolling_mean_10 = yahoo_data['yahoo_price'].rolling(window=10).mean()

```
Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively
```
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```
Plot Moving Average
```
plt.figure(figsize=(12, 6))
plt.plot(yahoo_data['yahoo_price'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of yahoo price Data')
plt.xlabel('Date')
plt.ylabel('Yahoo price')
plt.legend()
plt.grid()
plt.show()


```

Perform data transformation to better fit the model
```
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
yahoo_data = data[['yahoo_price']]



data_monthly = data.resample('MS').sum()
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten()
scaled_data = pd.Series(scaled_array, index=data_monthly.index)

```
Exponential Smoothing
```
# The data seems to have additive trend and multiplicative seasonality
scaled_data=scaled_data+1
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

# Make predictions on the test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot the training data, test data, and predictions
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
plt.show() # Added plt.show() to display the plot

# Calculate RMSE and variance/mean of scaled data
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()
```
Make predictions for one fourth of the data
```
model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12)
model_fit = model.fit()
predictions = model_fit.forecast(steps=int(len(data_monthly)/4))

ax=data_monthly.plot() # Added seasonal_periods
predictions.plot(ax=ax)
ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Months') 
ax.set_ylabel('yahoo price')
ax.set_title('Prediction')
plt.show() # Added plt.show() to display the plot

```
```

### OUTPUT:

Original data:

![image](https://github.com/user-attachments/assets/644b20e1-7d22-444f-aa0c-2e05c10d9627)







Moving Average:- (Rolling)

window(5):

![image](https://github.com/user-attachments/assets/40deee39-606c-4c60-801b-e022be6e37c5)




window(10):

![image](https://github.com/user-attachments/assets/9f281c8e-7cce-4f05-91d9-25e01291095a)


plot:


![image](https://github.com/user-attachments/assets/8dc3a651-0b32-4a77-9a77-8554db5e72a9)

Exponential Smoothing:-

Test:

![image](https://github.com/user-attachments/assets/213cf8f6-f27c-4499-a3ed-24700404d351)


Performance:

![image](https://github.com/user-attachments/assets/213cf8f6-f27c-4499-a3ed-24700404d351)


Prediction:

![image](https://github.com/user-attachments/assets/42662879-6142-4783-8594-303055f3833f)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
