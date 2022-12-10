import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib as plt
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
#lat, long y zipcode no nos influye ya que estan todas las casas en la misma zona de Seattle, Tacoma y alrededores (hemos buscado en google maps)


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

#Ahora vamos a quitar las columnas que no nos hacen falta: lat, long, date, zipcode

df = df.drop(columns = ['date', 'lat', 'long', 'zipcode'], axis = 1)


#Para simplificar el análisis, vamos a agrupar todas las columnas que tengan las maediciones de metros cuadrados, haciendo una única columna que diga todos los metros cuadrados de la casa.
#Tendremos al final dos columnas: la medida del terreno (lot) y la medida de la casa (living)
#No nos interesan las medidas anteriores a las reformas ya que no existen. 
#above y basement los quitamos ya que living es la suma de estas, y en el precio influyen los metos cuadrados totales
df = df.drop(columns = ['sqft_above', 'sqft_basement', 'sqft_living', 'sqft_lot'], axis = 1)
print(df.head())
print(df.columns)

#Como hay muy pocas casas renovadas, hay muchos ceros en la columna renovate. 
#Por tanto, nos podemos quitar esa columna si en el año de construcción ponemos el año de renovacion de las pocas que se han renovado

for i in range(len(df['yr_renovated'])):
    if df['yr_renovated'][i] != 0:
        df['yr_built'][i] = df['yr_renovated'][i]

df = df.drop(columns = ['yr_renovated'], axis = 1)
print(df['yr_built'])

#Ahora que el dataset está limpio, vamos a pasar a hacer la regresión
#Primero, vamos a entrenar y testear.

df.corr()

y = df['price']
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logreg = LogisticRegression(max_iter= 2000)

logreg.fit(x_train, y_train)
scoretrain = logreg.score(x_train, y_train)
scoretest = logreg.score(x_test, y_test)

precision_train = accuracy_score(y_train, logreg.predict(x_train))
precision_test = accuracy_score(y_test, logreg.predict(x_test))

sns.headmap(confusion_matrix(y_train, logreg.predict(x_train), annot= True))
plt.title('MATRIZ TRAIN')
plt.show()

sns.headmap(confusion_matrix(y_test, logreg.predict(x_test), annot= True))
plt.title('MATRIZ TEST')
plt.show()