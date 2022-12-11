import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns


df1 = pd.read_csv('data/casas_limpio.csv', delimiter= ',')

#Standart scale
#Transformamos los datos en una scale para poder pasarlos a pickle file
ss = StandardScaler()
df_transformado = ss.fit_transform(df1)
#print(df_transformado)
x_transformado = df_transformado[:, 0]
y_transformado = df_transformado[:, 1]
sns.scatterplot(x_transformado, y_transformado)