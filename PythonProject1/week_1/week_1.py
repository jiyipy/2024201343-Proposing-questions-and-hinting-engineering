import pandas as pd
import numpy as np


df_1 = pd.read_excel("data/第一周作业/score.xlsx")
df_2 = pd.read_excel("data/第一周作业/studdent.xlsx")
df = pd.merge(df_1, df_2, on="学号")
gender_avg = df.groupby("性别").agg({"成绩": np.mean})
gender_subject_avg = df.groupby(["性别","科目"]).agg({"成绩": np.mean})
# ---------目标1-------------
print(f"目标1:\n{gender_avg}")
# ---------目标2-------------
print(f"目标2:\n{gender_subject_avg}")
# ---------目标3-------------
total_scores = df.groupby('学号').agg(
    总成绩=('成绩', 'sum'),
    性别=('性别', 'first')
).reset_index()

# 计算性别维度的方差
gender_variance = total_scores.groupby('性别')['总成绩'].var().reset_index()
gender_variance.columns = ['性别', '方差']

# 结果输出
print("目标3:\n每个学生的总成绩：")
print(total_scores[['学号', '性别', '总成绩']])
print("\n男女总成绩方差对比：")
print(gender_variance)
female_var = gender_variance[gender_variance['性别'] == '女']['方差'].values[0]
male_var = gender_variance[gender_variance['性别'] == '男']['方差'].values[0]
if female_var < male_var:
    print("女生总成绩更稳定（方差更小）")
elif female_var > male_var:
    print("男生总成绩更稳定（方差更小）")
else:
    print("男女成绩稳定性相同")
# df_2.to_html('./data/df2.html')