# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing and preprocessing data
data_raw = pd.read_csv('/Users/marcusring/Desktop/KU/Tredje år/DatFin/Python/Bahcelor_GARCH/priceDatacopy.csv', sep=';')
df_prices = data_raw.dropna(subset=['Spot.price', 'hour', 't'])
df_prices = df_prices[df_prices['Spot.price'] > 0]
df_prices['hour'] = pd.to_numeric(df_prices['hour'], errors='coerce')
df_prices['t'] = pd.to_datetime(df_prices['t'], errors='coerce')
df_prices = df_prices.dropna(subset=['t'])

# Hourly means
h_means = df_prices.groupby('hour')['Spot.price'].mean().sort_index()

# Plotting hourly average
plt.figure(figsize=(10, 6))
plt.plot(h_means.index, h_means.values, marker='o', linestyle='-', color='b')
plt.title('Average electricity spot price by hour')
plt.xlabel('Hour of day')
plt.ylabel('Average spot price')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.grid(True)
plt.show()

# Printing hourly means
print("Hourly average spot prices:")
print(h_means)

# Filtering on December 24^th 2023
date_filter = pd.to_datetime('2022-12-24')
df_dec24 = df_prices[df_prices['t'].dt.date == date_filter.date()]

# Ploting spot price for Dec 24^th 2023
plt.figure(figsize=(10, 6))
plt.plot(df_dec24['hour'], df_dec24['Spot.price'], marker='o', linestyle='-', color='blue')
plt.title('Electricity spot price on December 24th 2022')
plt.xlabel('Hour of day')
plt.ylabel('Spot price')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.grid(True)
plt.show()

# Showing seasonality in hourly returns
# Calculating log returns
df_prices['log_return'] = np.log(df_prices['Spot.price'] / df_prices['Spot.price'].shift(1))

# Dropping N/A log returns
df_returns = df_prices.dropna(subset=['log_return'])

# Hourly means
hourly_return_means = df_returns.groupby('hour')['log_return'].mean().sort_index()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(hourly_return_means.index, hourly_return_means.values, marker='o', linestyle='-', color='blue')
plt.title('Average return by hour')
plt.xlabel('Hour of day')
plt.ylabel('Average return')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.grid(True)
plt.show()



