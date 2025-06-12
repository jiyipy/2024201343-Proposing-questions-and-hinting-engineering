import numpy as np
import pandas as pd

df=pd.read_csv('../data/beijing_tianqi_2019.csv')
df['date']=pd.to_datetime(df['ymd'])
df = df.set_index('date')
d=df.resample('3D')['aqi'].agg(np.mean)
print(d)