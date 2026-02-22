import gridstatus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

iso = gridstatus.Ercot()

end_date = pd.Timestamp.now(tz = 'US/Central')
start_date = end_date - pd.Timedelta(days = 365)

df_load = iso.get_load(start=start_date, end=end_date)

df_load = df_load[['Interval Start', 'Load']].rename(columns={'Interval Start': 'timestamp', 'Load': 'load_mw'})

print(df_load.head())

all_fuel_data = []
for single_date in pd.date_range(start=start_date, end=end_date, freq='D'):
  try:
    daily_fuel = iso.get_fuel_mix(date=single_date)
    if daily_fuel is not None:
      all_fuel_data.append(daily_fuel)
  except Exception as e:
    print(f"Error fetching data for {single_date.strftime('%Y-%m-%d')}: {e}")

df_fuel = pd.concat(all_fuel_data).reset_index(drop=True)
df_fuel_pivot = df_fuel.rename(columns={'Time': 'timestamp'})
df_fuel_pivot.columns = [c.lower() for c in df_fuel_pivot.columns]

print(df_fuel_pivot.head())

df_load['timestamp'] = pd.to_datetime(df_load['timestamp'])
df_fuel_pivot['timestamp'] = pd.to_datetime(df_fuel_pivot['timestamp'])

# Resample df_load to 5-minute frequency, taking the mean
df_load_5min = df_load.set_index('timestamp').resample('5T').mean().reset_index()

# Merge df_load_5min and df_fuel_pivot into df (df_fuel_pivot is already 5-minute data)
df = pd.merge(df_load_5min, df_fuel_pivot, on='timestamp', how='outer')

df = df.interpolate(method='linear')
df = df.dropna()

if 'solar' in df.columns and 'wind' in df.columns:
 df['renewable_gen'] = df['solar'] + df['wind']
 df['net_load'] = df['load_mw'] - df['renewable_gen']
else:
 raise ValueError("Wind/Solar data missing from API response")

df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek

for lag in [1, 24, 168]:
 df[f'lag_{lag}'] = df['net_load'].shift(lag)

df = df.dropna()

print("Training Model...")

split_idx = int(len(df)* 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Length of df: {len(df)}")
print(f"Length of train: {len(train)}")
print(f"Length of test: {len(test)}")

feature = ['hour', 'month', 'day_of_week', 'lag_1', 'lag_24', 'lag_168', 'solar', 'wind']
target = 'net_load'

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1)

model.fit(train[feature], train[target],
     eval_set=[(test[feature], test[target])],
     verbose=False)

preds = model.predict(test[feature])
rmse = np.sqrt(mean_squared_error(test[target], preds))
r2 = r2_score(test[target], preds)

print(f"\n--- Results on Texas Grid (Test Set) ---")
print(f"RMSE: {rmse:.2f} MW")
print(f"R2 Score: {r2:.4f}")

plt.figure(figsize=(15, 7))
subset = 168 # View 1 week
plt.plot(test['timestamp'].iloc[:subset], test['net_load'].iloc[:subset], label='Actual ERCOT Net Load', color='black', alpha=0.6)
plt.plot(test['timestamp'].iloc[:subset], preds[:subset], label='XGBoost Forecast', color='#D43F3A', linestyle='--')
plt.title(f'Texas Grid Net-Load Forecast (R2: {r2:.2f})', fontsize=16)
plt.ylabel('Power (MW)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
importance = pd.DataFrame({'feature': feature, 'importance': model.feature_importances_})
print("\nTop Drivers of Net Load:")
print(importance.sort_values('importance', ascending=False))