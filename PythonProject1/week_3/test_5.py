import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df=pd.read_csv('../data/beijing_tianqi_2019.csv')
# 转换日期
df['date']=pd.to_datetime(df['ymd'])
# 按月分组统计aqi的均值
df_aqi_m=df.groupby(df['date'].dt.month).agg({'aqi':np.mean})
print(df_aqi_m)
# 绘制折线图
plt.plot(df_aqi_m.index,df_aqi_m.aqi)
plt.show()

df_aqi_q=df.groupby(df['date'].dt.quarter).agg({'aqi':np.mean})
print(df_aqi_q)
# 绘制柱状图
plt.bar(df_aqi_q.index,df_aqi_q.aqi)
plt.show()