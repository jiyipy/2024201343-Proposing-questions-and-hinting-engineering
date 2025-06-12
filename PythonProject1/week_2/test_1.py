import numpy as np
import pandas as pd

# ------------------------------
# 创建一个python原声列表
# list_1 = [1,2,3]
# a_1 = np.array([list_1,list_1,list_1])
# print(list_1)
# print(a_1)


# -------------------------------
# 创建指定范围数组
# a_2 = np.arange(0,100,2)
# a_3 = np.array([a_2,a_2])
# print(a_3)

# -------------------------------
# linespace等差序列数组
# a_4 = np.linspace(0,1001,6)
# print(a_4)

# ---------------------------------
# a_5 = np.array([[10,20,30],[40,50,60]])
# a_6 = a_5[-1][1]
# print(a_6)
# ----------------------------------
# b = np.arange(1,10)
# a_7 = b.reshape(3,3)
# a_8 = a_7.reshape(-1)
# print(a_8)
# print(np.sum(a_7,axis=1))

# ------------------------
excel_1 = pd.read_excel("../data/中国数字经济.xlsx")
print(excel_1)
print(excel_1.isnull().sum())


# excel_1.dropna(inplace=True,axis=1,how="all")


# excel_1.fillna({"互联网普及率":0,"数据来源":"缺失"},inplace=True)
# print(excel_1)


# excel_1.fillna(method="ffill",inplace=True)
# print(excel_1)

excel_1["互联网普及率"].fillna(excel_1["互联网普及率"].mean())
print(excel_1)