import numpy as np
import pandas as pd


student = pd.read_excel("../data/studdent.xlsx")
student_1 = pd.read_excel("../data/studdent_1.xlsx")
# print(student)
score = pd.read_excel("../data/score.xlsx")
df_2019 = pd.read_csv("../data/beijing_tianqi_2019.csv")
df_2018 = pd.read_csv("../data/beijing_tianqi_2018.csv",encoding='gbk')

# # on,left_on,right_on参数
# df = pd.merge(student_1,score,left_on="学生编号",right_on="学号")
# print(df)


# # 连接方式
# df_1 = pd.merge(student,score,on="学号",how="left")
# print(df_1)

# df = pd.concat([df_2018,df_2019],axis=0)
# print(df)

df=pd.DataFrame(
    {
        'item':['item0','item0','item1','item1'],
        'CType':['Gold','Bronze','Gold','Sliver'],
        'USD':[1,2,3,4],
        'EU':[1,2,3,4]

    }
)

# df_1 = df.pivot(index="item",columns="CType",values="USD")
# print(df_1)

df_3 = df.stack()
print(df_3)

df_4 = df.unstack()
print(df_4)

df_5 = df.unstack()
print(df_5)