import pandas as pd
import numpy as np
# 1. 读取Excel文件
file1_path = 'C:/Users/han/Desktop/dataset/GDP.xlsx'
file2_path = 'C:/Users/han/Desktop/dataset/65.xlsx'
file3_path = 'C:/Users/han/Desktop/dataset/64.xlsx'
file4_path = 'C:/Users/han/Desktop/dataset/Annual_growth_rate.xlsx'
data1 = pd.read_excel(file1_path, header=None)
data2 = pd.read_excel(file2_path, header=None)
data3 = pd.read_excel(file3_path, header=None)
data4 = pd.read_excel(file4_path, header=None)
# 2. 提取第一行（年份）和第二行（GDP）
years = data1.iloc[0, 0:]  # 年份数据从第1列开始
gdp_values = data1.iloc[1, 0:]  # GDP数据从第1列开始
sixty_five_values = data2.iloc[1, 0:]
sixty_four_values = data3.iloc[1, 0:]
annual_growth_rate= data4.iloc[1, 0:]
# 3. 创建一个包含年份和GDP的数据框
all_data = pd.DataFrame({
    'Year': years.str.extract(r'(\d+)')[0].astype(int),  # 提取年份，使用原始字符串避免转义错误
    'GDP': gdp_values.astype(float),  # 将GDP转换为浮点型
    '65': sixty_five_values.astype(float),
    '64': sixty_four_values.astype(float),
    'rate': annual_growth_rate.astype(float)
})

# 4. 显示处理好的数据
print(all_data.head())