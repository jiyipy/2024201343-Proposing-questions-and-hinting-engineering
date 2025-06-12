import numpy as np
import pandas as pd

df = pd.read_csv("../data/航空公司.csv",encoding="gb18030")
print(df)
# print(type(df["FFP_DATE（入会时间）"][0]))
df["date"] = pd.to_datetime(df["FFP_DATE（入会时间）"])

df['year'] = df['date'].dt.year #提取年份
df['month'] = df['date'].dt.month #提取月份
df['quarter']= df['date'].dt.quarter #提取季度
df['day']= df['date'].dt.day # 提取日期
df['day_of_year'] = df['date'].dt.dayofyear # 一年中的第几天
df['is_weekend'] = df['date'].dt.weekday < 5  # 是否工作日、周一到周五为工作日
df["week"] = df["date"].dt.isocalendar().week #提取第几周
df['year_month'] = df['date'].dt.strftime('%Y-%m') #提取年和月
print(df)