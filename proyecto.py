import pandas as pd

#Leemos con pandas el csv
df = pd.read_csv('regression_data.csv', delimiter = ';')
print(df.head())