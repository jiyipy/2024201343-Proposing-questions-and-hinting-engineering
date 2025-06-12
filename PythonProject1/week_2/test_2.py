import numpy as np
import pandas as pd


df = pd.read_excel("../data/中国数字经济.xlsx")
print(df)

print(df.duplicated(subset=["地区","互联网普及率"]).sum)

df.drop_duplicates(inplace=True)

print(df.duplicated().sum)
