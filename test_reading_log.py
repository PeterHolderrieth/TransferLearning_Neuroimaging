import pandas as pd 

df = pd.read_csv('test_table.log', header=None)
print(df.head())