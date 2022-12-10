import pandas as pd

#Leemos con pandas el csv
df = pd.read_csv('regression_data.csv', delimiter = ';')
print(df.head())

#Examinamos el csv
print('COLUMNAS')
print(df.columns)
print('\n')
print('INFO')
df.info()

#Observación: columnas no numércicas: date, bathrooms(arreglar), floors(arreglar), lat(arreglar), long(arreglar)
# En principio no se ve ningún dato nulo  

#Separamos variables numéricas y categóricas 