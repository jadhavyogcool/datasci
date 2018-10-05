# Import the Libraries

import pandas as pd
import numpy as np

dataset = pd.read_csv('1_WaterWellDev.csv')
data = dataset.values

print(dataset.info())
print(dataset.tail())
print(dataset.isnull())

dataset.fillna(5)
no_na = dataset.dropna(axis=0)
no_na = no_na.reset_index()

new_df = no_na.iloc[0:, 1:]
new_df.to_csv('new_wellWaterDev.csv', index=False)