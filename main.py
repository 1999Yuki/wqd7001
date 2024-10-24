import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

df5 = pd.DataFrame({
    'Year': years5.str.extract(r'(\d+)')[0].astype(int),
    'Healthcare': pd.to_numeric(healthcare_values, errors='coerce')
})

# 4. Filter data to keep only rows with years 2000-2023
df1_filtered = df1[df1['Year'].between(2000, 2023)]
df5_filtered = df5[df5['Year'].between(2000, 2023)]

# 5. Merge the two DataFrames based on the 'Year' column
merged_data = pd.merge(df1_filtered, df5_filtered, on='Year', how='inner')

# 6. Custom fill for 2022 and 2023
merged_data.loc[merged_data['Year'] == 2022, 'Healthcare'] = 4.48
merged_data.loc[merged_data['Year'] == 2023, 'Healthcare'] = 4.55

# Check for null values
print(merged_data.isnull().sum())

# 7. Display processed data
print(merged_data)

# 8. Normalization
# Ensure no columns have the same max and min values
if (merged_data.max() - merged_data.min()).eq(0).any():
    print("Warning: One or more columns have no variation. Adjust normalization accordingly.")
else:
    df_normalized = (merged_data - merged_data.min()) / (merged_data.max() - merged_data.min())

# Plot data
plt.figure(figsize=(10, 6))
for column in df_normalized.columns[1:]:  # Skip 'Year' column for plotting
    plt.plot(merged_data['Year'], df_normalized[column], label=column)
plt.xticks(np.arange(2000, 2024, 2))
plt.legend()
plt.title('Normalized Values (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Normalized Value')
plt.grid()
plt.show()
