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
healthcare_values = data5.iloc[1, 0:]

# 6. Custom fill for 2022 and 2023
healthcare_values.iloc[22] = 4.48  # Assuming 2022 is at index 22
healthcare_values.iloc[23] = 4.55  # Assuming 2023 is at index 23

# 3. Create DataFrames containing years and values
df1 = pd.DataFrame({
    'Year': years1.str.extract(r'(\d+)')[0].astype(int),
    'GDP(10million)': pd.to_numeric(gdp_values.astype(float) / 10000000, errors='coerce'),
    '>65': pd.to_numeric(sixty_five_values.astype(float), errors='coerce'),
    '15-64': pd.to_numeric(sixty_four_values.astype(float), errors='coerce'),
    'Annual_growth': pd.to_numeric(annual_growth_rate.astype(float), errors='coerce'),
    'Healthcare': pd.to_numeric(healthcare_values, errors='coerce')
})

# Check for null values
print(df1.isnull().sum())

# 7. Display processed data
print(df1)

# 8. Normalization
# Ensure no columns have the same max and min values
if (df1.max() - df1.min()).eq(0).any():
    print("Warning: One or more columns have no variation. Adjust normalization accordingly.")
else:
    df_normalized = (df1 - df1.min()) / (df1.max() - df1.min())

    # Plot data
    plt.figure(figsize=(10, 6))
    for column in df_normalized.columns[1:]:  # Skip 'Year' column for plotting
        plt.plot(df1['Year'], df_normalized[column], label=column)

    plt.xticks(np.arange(2000, 2024, 2))
    plt.legend()
    plt.title('Normalized Values (2000-2023)')
    plt.xlabel('Year')
    plt.ylabel('Normalized Value')
    plt.grid()
    plt.show()
