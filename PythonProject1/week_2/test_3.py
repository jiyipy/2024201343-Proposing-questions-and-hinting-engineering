import numpy as np
import pandas as pd


excel_1 = pd.read_excel("../data/中国数字经济.xlsx")
# print(excel_1)
#
excel_1.fillna({"互联网普及率":excel_1["互联网普及率"].mean(),"数据来源":"官网"},inplace=True)
print(excel_1)
#
#
# # 查询
# excel_2 = excel_1.loc[(excel_1["数字普惠金融指数"]>100)&(excel_1["互联网普及率"]<70),:]
#
# print(excel_2)
#
# excel_3 = excel_1.iloc[2:6,1:3]
# print(excel_3)
#
# excel_1.loc[excel_1["年份"]>2013,"互联网普及率"] = 10000
# print(excel_1)
# df = pd.DataFrame({"A":[1,2],"B":[3,4]})
# df.loc[len(df)] = [5,6]
# print(df)
# 新添加一行总指标,为互联网相关产出乘以移动互联网人数
# excel_1["总指标"] = excel_1["互联网相关产出"] * excel_1["移动互联网用户数"]
# # print(excel_1)
# def new_list(x):
#     if x["互联网普及率"] >= 50:
#         return "高"
#     else:
#         return "低"
#
# excel_1["信息化程度"] = excel_1.apply(new_list,axis=1)
# print(excel_1)

excel_2 = excel_1.groupby("地区").agg({"互联网相关从业人数":np.sum})
excel_2.sort_values("互联网相关从业人数",ascending=False,inplace=True)
print(excel_2)
excel_3 = excel_1.groupby("地区").agg({"互联网相关从业人数":[np.sum,np.median],"数字普惠金融指数":[np.max,np.std]})
print(excel_3)
excel_3.to_excel("1.xlsx")