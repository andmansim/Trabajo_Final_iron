import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Leemos con pandas el csv
df = pd.read_csv('regression_data.csv', delimiter = ';')
print(df.head())

#Examinamos el csv
print('COLUMNAS')
print(df.columns)
print('\n')
print('INFO')
df.info()

#Observación: 
#columnas no numércicas: date, bathrooms(arreglar), floors(arreglar), lat(arreglar), long(arreglar)
#En principio no se ve ningún dato nulo  
#Como queremos predecir el precio, la columna que va a ir por separado es price
#Para predecir el precio necesitamos: bedrooms, bathrooms, sqft (los examinamos para ver si los podemos agrupar), floors, condiciones, grade, year_built, year_renovate, waterview
#lat, long no nos influye ya que estan todas las casas en la misma zona de Seattle, Tacoma y alrededores (hemos buscado en google maps)


#Pasamos a la limpieza de los datos

#Hacemos copia de los datos
df_copia = df.copy()
#Los datos que tengan comas que no tengan sentido, como 2,25 baños, vamos a pasarlos a enteros:

def redondear(columna):
    for i in range(len(df)):
        if ',' in df[columna][i]:
            df[columna][i] = float(df[columna][i].replace(',', '.'))
            df[columna][i] = round(df[columna][i])
    return df

redondear('bathrooms')
print(df['bathrooms'])
redondear('floors')
print(df['floors'])