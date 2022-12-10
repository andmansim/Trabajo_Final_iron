import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pylab as plt
import statsmodels.api as sm
import numpy as np
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

'''redondear('bathrooms')
print(df['bathrooms'])
redondear('floors')
print(df['floors'])
'''
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

#Las columnas que tengan que ver con numero de habitaciones, las podemos agrupar para simpllificar los calculos
'''df['habitaciones'] = 0
for i in range(len(df['bedrooms'])):
       df['habitaciones'][i] = int(df['bedrooms'][i]) + int(df['bathrooms'][i])'''

#Ahora que el dataset está limpio, vamos a pasar a hacer la regresión
#Primero, vamos a entrenar y testear.

print(df.corr())

#Por la matriz de correlacion vemos que los unicos datos con los que el precio puede estar más relacionado es con los 
#metros cadrados de vivienda y con la calidad (sqft_living15, grade)
#Por tanto, vamos a probar la recta de regresión con esas dos columnas

x2 = df['price']

y6 = df['grade']
y7 = df['sqft_living15']

x2_const = sm.add_constant(x2)

def regresion(y, x, xconst):  
    modelo = sm.OLS(y, xconst).fit()
    pred = modelo.predict(xconst)
    try:
        const = modelo.params[0]
        coef = modelo.params[1]
        x_l = np.linspace(x.min(), x.max(), 50)
        y_l = coef*x_l + const
    except:
        pass

    plt.plot(x_l, y_l, label = f'{x.name} vs {y.name} = {coef}*{x.name} + {const}')
    plt.scatter(x, y, marker = 'x', c = 'g', label = f'{x.name} vs {y.name}')
    plt.title('regresion lineal')
    plt.xlabel(f'{x.name}')
    plt.ylabel(f'{y.name}')
    plt.show()

regresion(y6, x2, x2_const)
regresion(y7, x2, x2_const)
print(df.groupby(['price']).count())
#Se observa en las gráficas que la recta está descuadrada debido a los valores atípicos. Vamos a eliminar estos valores y a volverlo a intentar
print(df['grade'].unique()) #El 12 y el 13 aparecen muy poco, son filas que podemos eliminar
for i in range(len(df['grade'])):
    if df['grade'][i] == 12 or df['grade'][i] == 13:
        df.drop([i], axis = 0, inplace = True )
        df = df.reset_index()

for i in range(len(df['price'])):
    if df['price'][i] < 7000000:
        pass
    else:
        df.drop([i], axis = 0, inplace = True )
        df = df.reset_index()


a = df['sqft_living15'].unique()


print(df.groupby(['sqft_living15']).count()) #Como hay muchísimos valores únicos, vamos a agrupar por rangos
'''for i in range(len(df['sqft_living15'])):
    if int(df['sqft_living15'][i]) < 600:
        df = df.drop([i], axis = 0)
    if  600 <= int(df['sqft_living15'][i]) < 800:
        df['sqft_living15'][i] = 700
    if 800 <= int(df['sqft_living15'][i]) < 1000:
        df['sqft_living15'][i] = 900
    if 1000 <= int(df['sqft_living15'][i]) < 1500:
        df['sqft_living15'][i] = 1250
    if 1500 <= int(df['sqft_living15'][i]) < 2000:
        df['sqft_living15'][i] = 1750
    if 2000 <= int(df['sqft_living15'][i]) < 3000:
        df['sqft_living15'][i] = 2500
    if 3000 <= int(df['sqft_living15'][i]) < 4000:
        df['sqft_living15'][i] = 3500
    if 4000 <= int(df['sqft_living15'][i]) < 5000:
        df['sqft_living15'][i] = 4500
    if 5000 <= int(df['sqft_living15'][i]) < 6000:
        df['sqft_living15'][i] = 5500
    if 6000 <= int(df['sqft_living15'][i]):
        df['sqft_living15'][i] = 6000
'''
x2 = df['price']

y6 = df['grade']
y7 = df['sqft_living15']

x2_const = sm.add_constant(x2)
regresion(y6, x2, x2_const)
regresion(y7, x2, x2_const)


'''y = df['price']
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logreg = LogisticRegression(max_iter= 2000)

logreg.fit(x_train, y_train)
scoretrain = logreg.score(x_train, y_train)
scoretest = logreg.score(x_test, y_test)

precision_train = accuracy_score(y_train, logreg.predict(x_train))
precision_test = accuracy_score(y_test, logreg.predict(x_test))

sns.heatmap(confusion_matrix(y_train, logreg.predict(x_train), annot= True))
plt.title('MATRIZ TRAIN')
plt.show()

sns.heatmap(confusion_matrix(y_test, logreg.predict(x_test), annot= True))
plt.title('MATRIZ TEST')
plt.show()
'''

#Vamos a hacer gráficas para entender como funcionan los datos, con respecto al preio

'''def puntos(xeje, yeje):
    sns.scatterplot(data=df, x = xeje, y = yeje)
    plt.show()

puntos('price', 'yr_built')
puntos('price', 'condition')
puntos('price', 'grade')
puntos('price', 'sqft_living15')
puntos('price', 'sqft_lot15')
'''
#Por último, construimos la recta de regresion lineal
#La recta se construye con dos variables, por lo que hay que construir una para cada parcon price
