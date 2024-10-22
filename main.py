import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. 读取Excel文件
file1_path = 'dataset/GDP.xlsx'
file2_path = 'dataset/65.xlsx'
file3_path = 'dataset/64.xlsx'
file4_path = 'dataset/Annual_growth_rate.xlsx'
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
    'GDP(10million)': gdp_values.astype(float)/10000000,  # 将GDP转换为浮点型
    '>65': sixty_five_values.astype(float),
    '15-64': sixty_four_values.astype(float),
    'annual_growth': annual_growth_rate.astype(float)
})

# 4. 显示处理好的数据
#print(all_data.head())
# 5. 归化
df = all_data
# 归一化数据到 [0, 1] 范围
df_normalized = (df.iloc[:, 1:] - df.iloc[:, 1:].min()) / (df.iloc[:, 1:].max() - df.iloc[:, 1:].min())

# 绘制数据
for column in df_normalized.columns:
    plt.plot(df['Year'], df_normalized[column], label=column)

plt.xticks(np.arange(1960, 2024, 10))  # 设置 x 轴刻度
plt.legend()
plt.title('Normalized value')
plt.xlabel('year')
plt.ylabel('Normalized value')
plt.grid()
plt.show()