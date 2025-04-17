import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

# Load your data
data_raw = pd.read_csv("priceData.csv", sep=";")
df_prices = data_raw["Spot.price"]
df_dates = data_raw["t"]
df_workday = data_raw["dayOfWeek"]

# Slice & interpolate prices
prices = np.abs(df_prices.values[23:232])
dates = df_dates.values[23:232]

s = pd.Series(prices)
prices = s.interpolate(method='linear', limit_direction='both').to_numpy()



#we define the seasonal period (24 hours)
seasonal_period = 24

# Fit SARIMA: (p,d,q)x(P,D,Q,s)
# Here’s a common starting point: SARIMA(1,1,1)x(1,1,1,24)
model = sm.tsa.SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
results = model.fit()

# Forecast 24 steps ahead
forecast = results.get_forecast(steps=24)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()  # This may return a NumPy array
lower = conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0]
upper = conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.abs(df_prices.values[23:232+24]), label='Observed')
plt.plot(np.arange(len(prices), len(prices)+24), forecast_mean, label='Forecast', linestyle='--')
plt.fill_between(np.arange(len(prices), len(prices)+24), lower, upper, color='gray', alpha=0.3)
plt.title('SARIMA Forecast (Next 24 Hours)')
plt.xlabel('Time (Hourly)')
plt.ylabel('Electricity Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()