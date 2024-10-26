import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Read Excel files
file1_path = 'dataset/GDP.xlsx'
file2_path = 'dataset/65.xlsx'
file3_path = 'dataset/64.xlsx'
file4_path = 'dataset/Annual_growth_rate.xlsx'
file5_path = 'dataset/healthcare.xlsx'

data1 = pd.read_excel(file1_path, header=None)
data2 = pd.read_excel(file2_path, header=None)
data3 = pd.read_excel(file3_path, header=None)
data4 = pd.read_excel(file4_path, header=None)
data5 = pd.read_excel(file5_path, header=None)

# 2. Extract the first row (years) and the second row (values) for all datasets
years1 = data1.iloc[0, 0:]
gdp_values = data1.iloc[1, 0:]
sixty_five_values = data2.iloc[1, 0:]
sixty_four_values = data3.iloc[1, 0:]
annual_growth_rate = data4.iloc[1, 0:]

years5 = data5.iloc[0, 0:]
healthcare_values = data5.iloc[1, 0:]

# 3. Create DataFrames containing years and values
df1 = pd.DataFrame({
    'Year': years1.str.extract(r'(\d+)')[0].astype(int),
    'GDP(10million)': pd.to_numeric(gdp_values.astype(float) / 10000000, errors='coerce'),
    '>65': pd.to_numeric(sixty_five_values.astype(float), errors='coerce'),
    '15-64': pd.to_numeric(sixty_four_values.astype(float), errors='coerce'),
    'Annual_growth': pd.to_numeric(annual_growth_rate.astype(float), errors='coerce')
})

# 4. Filter data to keep only rows with years 2000-2023
df_filtered = df1[df1['Year'].between(2000, 2023)]

# 5. Prepare data for GRU
# Select the '>65' column for prediction
data = df_filtered['>65'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create a dataset with a window of past values
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 3  # Use the last 3 years to predict the next year
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 6. Build the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(50))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# 7. Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# 8. Make predictions for the next 10 years
predictions = []
last_data = scaled_data[-time_step:].reshape(1, time_step, 1)

for _ in range(10):
    next_prediction = model.predict(last_data)
    predictions.append(next_prediction[0, 0])  # Store the prediction
    # Update last_data to include the new prediction
    last_data = np.concatenate((last_data[:, 1:, :], next_prediction.reshape(1, 1, 1)), axis=1)

# Inverse transform the predictions to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 9. Prepare future years for visualization
future_years = np.arange(2024, 2034)

# 10. Calculate RMSE and MAE for the predictions
# Get historical data for the last few years for comparison
historical_data = df_filtered['>65'].values[-len(predictions):]
rmse = np.sqrt(mean_squared_error(historical_data, predictions[:len(historical_data)]))
mae = mean_absolute_error(historical_data, predictions[:len(historical_data)])

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# 11. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df_filtered['Year'], df_filtered['>65'], label='Historical >65 Population', marker='o')
plt.plot(future_years, predictions, label='Predicted >65 Population (2024-2033)', marker='o')
plt.title('Population Aged >65 Prediction')
plt.xlabel('Year')
plt.ylabel('Population')
plt.xticks(np.arange(2000, 2034, 2))
plt.legend()
plt.grid()
plt.show()
