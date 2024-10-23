import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Read Excel files
file1_path = 'dataset/GDP.xlsx'
file2_path = 'dataset/65.xlsx'
file3_path = 'dataset/64.xlsx'
file4_path = 'dataset/Annual_growth_rate.xlsx'
data1 = pd.read_excel(file1_path, header=None)
data2 = pd.read_excel(file2_path, header=None)
data3 = pd.read_excel(file3_path, header=None)
data4 = pd.read_excel(file4_path, header=None)

# 2. Extract the first row (years) and the second row (GDP)
years = data1.iloc[0, 0:]  # Year data starts from the first column
gdp_values = data1.iloc[1, 0:]  # GDP data starts from the first column
sixty_five_values = data2.iloc[1, 0:]
sixty_four_values = data3.iloc[1, 0:]
annual_growth_rate = data4.iloc[1, 0:]

# 3. Create a DataFrame containing years and GDP
all_data = pd.DataFrame({
    'Year': years.str.extract(r'(\d+)')[0].astype(int),  # Extract year
    'GDP(10million)': gdp_values.astype(float)/10000000,  # Convert GDP to float
    '>65': sixty_five_values.astype(float),
    '15-64': sixty_four_values.astype(float),
    'annual_growth': annual_growth_rate.astype(float)
})

# 4. Display processed data
# print(all_data.head())

# 5. Normalization
df = all_data
# Normalize data to [0, 1] range
df_normalized = (df.iloc[:, 1:] - df.iloc[:, 1:].min()) / (df.iloc[:, 1:].max() - df.iloc[:, 1:].min())

# Plot data
for column in df_normalized.columns:
    plt.plot(df['Year'], df_normalized[column], label=column)

plt.xticks(np.arange(1960, 2024, 10))  # Set x-axis ticks
plt.legend()
plt.title('Normalized value')
plt.xlabel('Year')
plt.ylabel('Normalized value')
plt.grid()
plt.show()