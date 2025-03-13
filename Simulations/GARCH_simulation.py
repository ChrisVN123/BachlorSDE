from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd

# Importing dataset
data = pd.read_csv('/Users/marcusring/Desktop/KU/Tredje år/DatFin/Python/Bahcelor_GARCH/priceDatacopy.csv',
                   delimiter=';')
data_filtered = data["Spot.price"].replace([0], np.nan).dropna()

# Check for negative or zero values before log transformation
if (data_filtered <= 0).any():
    print("Warning: Negative or zero values detected in Spot prices.")

logprices = np.log(data_filtered / data_filtered.shift(1)).dropna()

# Plot the returns
plt.figure(figsize=(10, 4))
plt.plot(logprices)
plt.title('Log Returns of Spot Prices', fontsize=20)
plt.show()

# PACF Plot
plot_pacf(logprices ** 2)
plt.show()

# Split data into training and test sets
n = len(logprices)
test_size = int(n * 0.1)
train, test = logprices[:-test_size], logprices[-test_size:]

# Fit the GARCH(1,1) model to avoid complexity for debugging
model = arch_model(train, p=1, q=1)
model_fit = model.fit(disp='off')  # Use 'lbfgs' method with maxiter
print(model_fit.summary())

# Predict volatility
predictions = model_fit.forecast(horizon=test_size)
plt.figure(figsize=(10, 4))
plt.plot(np.sqrt(predictions.variance.values[-1, :]), label="Predicted Volatility")
plt.plot(test.index, test, label="Actual Returns", alpha=0.6)
plt.title('Volatility Prediction', fontsize=20)
plt.legend()
plt.show()

# Rolling Forecast
rolling_predictions = []
for i in range(test_size):
    train_subset = logprices[:-(test_size - i)]
    model = arch_model(train_subset, p=1, q=1)  # Simplify to GARCH(1,1)
    model_fit = model.fit(disp='off')  # Use 'lbfgs' method with maxiter
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

plt.figure(figsize=(10, 4))
plt.plot(test.index, rolling_predictions, label="Rolling Forecast Volatility")
plt.plot(test.index, test, label="Actual Returns", alpha=0.6)
plt.title('Rolling Forecast Volatility Prediction', fontsize=20)
plt.legend()
plt.show()
