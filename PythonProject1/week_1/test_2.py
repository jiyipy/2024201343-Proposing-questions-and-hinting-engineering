import pandas as pd
import numpy as np


df = pd.read_csv("../data/航空公司.csv", encoding="gb18030")
# print(df)
df_1 = df.groupby("GENDER（性别）").agg({"AGE（年龄）":np.mean})
print(df_1)
df_2 = df.groupby(["GENDER（性别）","WORK_PROVINCE（工作地所在省份）"]).agg({"AGE（年龄）":np.mean})
print(df_2)
df_2.to_html('./data/df2.html')