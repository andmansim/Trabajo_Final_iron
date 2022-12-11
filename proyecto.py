import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pylab as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

#Leemos con pandas el csv
df = pd.read_csv('data/regression_data.csv', delimiter = ';')
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

#Las columnas que tengan que ver con numero de habitaciones, las podemos agrupar para simpllificar los calculos
df['habitaciones'] = 0
for i in range(len(df['bedrooms'])):
       df['habitaciones'][i] = int(df['bedrooms'][i]) + int(df['bathrooms'][i])

#Exportamos el csv limpio y organizado
df.to_csv('casas_limpio.csv')


#Standart scale
ss = StandardScaler()
df_transformado = ss.fit_transform(df)
df_transformado
x_transformado = df_transformado[:, 0]
y_transformado = df_transformado[:, 1]
sns.scatterplot(x_transformado, y_transformado)

#Ahora que el dataset está limpio, vamos a pasar a hacer la regresión
#Primero, vamos a entrenar y testear.

print(df.corr())

#Por la matriz de correlacion vemos que los unicos datos con los que el precio puede estar más relacionado es con los 
#metros cuadrados de vivienda y con la calidad (sqft_living15, grade)
#Por tanto, vamos a probar la recta de regresión con esas dos columnas

x2 = df['price']

y6 = df['grade']
y7 = df['sqft_living15']
y8 = df['habitaciones']


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
regresion(y8, x2, x2_const)
print(df.groupby(['price']).count())
#Se observa en las gráficas que la recta está descuadrada debido a los valores atípicos. Vamos a eliminar estos valores y a volverlo a intentar


print(df['grade'].unique()) #A partir del 10, aparecen muy poco, son filas que podemos eliminar
for i in range(len(df['grade'])):
    if df['grade'][i] > 9:
        df = df.drop([i], axis = 0)
x2 = df['price']

y6 = df['grade']

x2_const = sm.add_constant(x2)
regresion(y6, x2, x2_const)
#Vemos que, aún así, no nos sirve como referente para una regresión lineal, ya que la disposición de los datos no se acerca a una recta.

a = df['sqft_living15'].unique()
print(a)

y7 = df['sqft_living15']
regresion(y7, x2, x2_const)

y8 = df['habitaciones']
regresion(y8, x2, x2_const)
#Una vez quitadas esas filas, solucionamos el problema en las dos gráficas.
#La que más se puede acercar a una linea es la sqft_living15
#Por tanto, entrenaremos nuestro modelo con esa columna

y = df['price']
x = df.drop(['price','id', 'sqft_lot15', 'floors', 'waterfront', 'condition', 'yr_built', 'grade'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logreg = LogisticRegression(max_iter= 100)

logreg.fit(x_train, y_train)
scoretrain = logreg.score(x_train, y_train)
scoretest = logreg.score(x_test, y_test)

dicc = {'Train_score: ': scoretrain,
       'Test_score: ': scoretest}
print(dicc)

accuracy_score_train = accuracy_score(y_train, logreg.predict(x_train))
print('ACCURACY SCORE TRAIN')
print(accuracy_score_train)

accuracy_score_test = accuracy_score(y_test, logreg.predict(x_test))
print('ACCURACY SCORE TEST')
print(accuracy_score_test)

sns.heatmap(confusion_matrix(y_train, logreg.predict(x_train)), annot = True)
plt.title('MATRIZ CONFUSION DE TRAIN')
plt.show()
sns.heatmap(confusion_matrix(y_test, logreg.predict(x_test)), annot = True)
plt.title('MATRIZ CONFUSION DE TEST')
plt.show()

sns.heatmap(confusion_matrix(y_train, logreg.predict(x_train), annot= True))
plt.title('MATRIZ TRAIN')
plt.show()

sns.heatmap(confusion_matrix(y_test, logreg.predict(x_test), annot= True))
plt.title('MATRIZ TEST')
plt.show()

