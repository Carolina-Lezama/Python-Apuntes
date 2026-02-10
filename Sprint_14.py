
La algebra lineal es el núcleo de los algoritmos utilizados para crear modelos de machine learning

¿Qué es una Matriz?
tabla bidimensional, dispuesta en filas y columnas.
Matriz (3x2):
[2  5]
[1  3]
[4  7]
3 filas (horizontal)
2 columnas (vertical)
6 elementos en total

¿Qué es un Espacio Vectorial Multidimensional?
Progresión dimensional:
1D (línea): Solo necesitas 1 número
- Posición en una carretera: km 50

2D (plano): Necesitas 2 números  
- Ubicación en un mapa: (latitud, longitud)

3D (volumen): Necesitas 3 números
- Posición en una habitación: (largo, ancho, alto)

4D, 5D, 100D...: Necesitas 4, 5, 100 números
- En machine learning: cada característica es una dimensión

# (matriz 3x4)
personas = [
    [25, 60000, 175, 70],  # Juan
    [30, 75000, 180, 80],  # María  
    [28, 65000, 170, 65]   # Pedro
]

# Cada muestra de mineral es un punto en espacio 52D
muestra_1 = [210.8, 14.99, 8.08, 1.005, ...]  # 52 características
muestra_2 = [215.4, 14.98, 8.08, 0.990, ...]  # 52 características
Espacio vectorial: 52 dimensiones (una por cada característica)
Matriz de datos: 19,937 filas × 52 columnas








Cualquier dato numérico puede representarse como un vector.

En Python, los conjuntos de datos (a menudo) se representan como listas.
En matemáticas, un conjunto ordenado de datos (numéricos) es un vector o un vector aritmético

Cual operacion que se pueda realizar entre numeros se puede realizar entre vectores

# -------------Pasar de lista a array (Vector)----------------
import numpy as np

# Forma 1
numbers1 = [2, 3] # Lista de Python
vector1 = np.array(numbers1) # Matriz de NumPy

# Forma 2
vector2 = np.array([6, 2])

# -------------Pasar de array (Vector) a lista----------------
# Forma 1
numbers2 = list(vector2)

# Forma 2
list1 = vector1.tolist()

# --------------- Conversión: DataFrame → Vector NumPy --------------
# Un DataFrame de pandas
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
print(type(ratings['Price']))  # <class 'pandas.core.series.Series'>

ratings['Price'] = 
0     68
1     81
2     81
3     15
4     75
...
Name: Price, dtype: int64

# Un vector NumPy
price = ratings['Price'].values
print(type(price))  # <class 'numpy.ndarray'>

Resultado: array([68, 81, 81, 15, 75, 17, 24, 21, 76, 12, ...])

Diferencias clave:
Pandas Series:
- Tiene índices

NumPy Array:
- Solo datos puros

.values es como extraer solo los números sin los encabezados ni numeración, y ponerlos en una lista pura.

# ----------------- Determinar el tamaño del vector -----------------
print(len(vector2))

# ----------------- Array a Dataframe - extraccion de arrays -----------------
import pandas as pd

ratings_values = [ #cada par de [] representa una fila
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]

ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
print(ratings)
price = ratings['Price'].values   
quality = ratings['Quality'].values
print('Precio: ', price) #imprime array de valores
print('Calidad: ', quality)

visitors_count = len(ratings['Price'].values) #contar valores en el array
visitor4 = ratings.loc[4].values #extraer fila 4 como array 
print('Número de visitantes:', visitors_count)
print('Visitante 4:', visitor4)

vector_list = list(ratings.values) #convertir matriz en una lista de vectores
    #Resultado: 
    # [array([68, 18]), array([81, 19]), array([81, 22]), array([15, 75]), array([75, 15]), 
    # array([17, 72]), array([24, 75]), array([21, 91]), array([76,  6]), array([12, 74]), 
    # array([18, 83]), array([20, 62]), array([21, 82]), array([21, 79]), array([84, 15]), 
    # array([73, 16]), array([88, 25]), array([78, 23]), array([32, 81]), array([77, 35])]
print(vector_list)

# ----------------- Diferencia visual entre listas y arrays -----------------
Con comas: [75, 15] → esto es una lista de Python
Sin comas: [75 15] → esto es un array de NumPy

# ----------------- Visualizacion de vectores -----------------
Un vector bidimensional consta de 2 numeros que representan una posición en un plano XY.
El vector se representa mediante un punto o una flecha desde el origen (0,0) hasta la posición (x,y).
La flecha indica los dos componentes de un vector: magnitud y dirección

# Visualizacion con circulos en los vectores
import numpy as np
import matplotlib.pyplot as plt
vector1 = np.array([2, 3]) #puntos de coordenada
vector2 = np.array([6, 2])
plt.figure(figsize=(10, 10)) #tamaño de la figura
plt.axis([0, 7, 0, 7]) #establece los límites de los ejes x e y (rango de 0 a 7 en ambos ejes)
# El argumento 'ro' establece el estilo del gráfico
# 'r' - rojo
# 'o' - círculo
plt.plot([vector1[0], vector2[0]], [vector1[1], vector2[1]], 'ro') #plotea las posiciones
plt.grid(True) #cuadricula
plt.show()

# Visualizacion con flechas en los vectores
import numpy as np
import matplotlib.pyplot as plt
vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
plt.figure(figsize=(10, 10))
plt.axis([0, 7, 0, 7])
plt.arrow(
    0, #inicio de la flecha
    0,
    vector1[0], #posicion final de la flecha
    vector1[1],
    head_width=0.3, # controla el ancho de la punta de la flecha
    length_includes_head="True", # determina cómo se mide la longitud total de la flecha, true cuenta la cabeza en la medida
    color='b',
)
plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
plt.plot(0, 0, 'ro') #punto en el origen
plt.grid(True)
plt.show()

# ----------------- Visualizacion de datos reales como vectores -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])
plt.figure(figsize=(3.5,3.5))
plt.axis([0, 100, 0, 100])
price = ratings['Price'].values
quality = ratings['Quality'].values
plt.plot(price, quality, 'ro') #pasamos todas las filas
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.show()

# ----------------- Separacion de datos en listas -----------------

import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

visitors_1 = [] #contendran las caloraciones de los visitantes
visitors_2 = []
for visitor in list(ratings.values):
    # print(visitor) # muestra cada fila
    # print(visitor[0])
    # print(visitor[1])
    if visitor[0]<40 and visitor[1]>60:
        visitors_1.append(visitor)
    else:
        visitors_2.append(visitor)


print('Valoraciones de los visitantes del primer agregador:', visitors_1)
print('Valoraciones de los visitantes del primer agregador:', visitors_2)

# ----------------- Suma de vectores-----------------
# Del mismo tamaño
import numpy as np
vector1 = np.array([2, 3])
vector2 = np.array([6, 2])

# Sumar 2 vectores: sumar las coordenadas correspondientes de los vectores iniciales
sum_of_vectors = vector1 + vector2
print(sum_of_vectors)
# Resultado: [8 5]

# Dibujar plano con vectores:
plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6]) #establece los límites de los ejes x e y
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow3 = plt.arrow(
    0,
    0,
    sum_of_vectors[0],
    sum_of_vectors[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
arrow4 = plt.arrow(
    0,
    0,
    subtraction[0],
    subtraction[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow2, arrow3],
    ['vector1', 'vector2', 'sum_of_vectors', 'subtraction'],
    loc='upper left',
)
plt.grid(True)
plt.show()

# Visualzicacion de suma de 2 vectores junto con resultado final en otro vector
plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow2new = plt.arrow(
    vector1[0],
    vector1[1],
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow3 = plt.arrow(
    0,
    0,
    sum_of_vectors[0],
    sum_of_vectors[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow3],
    ['vector1', 'vector2', 'sum_of_vectors'],
    loc='upper left',
)
plt.grid(True)
plt.show()

# ----------------- Resta de vectores-----------------
# Del mismo tamaño
import numpy as np
vector1 = np.array([2, 3])
vector2 = np.array([6, 2])

# Restar 2 vectores: la diferencia entre coordenadas de los vectores iniciales
subtraction = vector2 - vector1
print(subtraction)
# Resultado: [4 -1]

plt.figure(figsize=(10.2, 10))
plt.axis([0, 8.4, -2, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow4 = plt.arrow(
    0,
    0,
    subtraction[0],
    subtraction[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
arrow1new = plt.arrow(
    vector2[0],
    vector2[1],
    -vector1[0], # Los signos negativos están invirtiendo la dirección del vector
    -vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow2, arrow4],
    ['vector1', 'vector2', 'subtraction'],
    loc='upper left',
)
plt.grid(True)
plt.show()

# ----------------- Ejemplo -----------------
import numpy as np
import pandas as pd

quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Capa de silicone para o iPhone 8',
    'Capa de couro para o iPhone 8',
    'Capa de silicone para o iPhone XS',
    'Capa de couro para o iPhone XS',
    'Capa de silicone para o iPhone XS Max',
    'Capa de couro para o iPhone XS Max',
    'Capa de silicone para o iPhone 11',
    'Capa de couro para o iPhone 11',
    'Capa de silicone para o iPhone 11 Pro',
    'Capa de couro para o iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity' : quantity_1}, index=models)

quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity' : quantity_2}, index=models)

vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2

stocks_united = pd.DataFrame( {'Quantity' : vector_of_quantity_united}, index=models)
print(stocks_united)

#  Resultado:                               Quantity
# Capa de silicone para o iPhone 8            107
# Capa de couro para o iPhone 8                87
# Capa de silicone para o iPhone XS           172
# Capa de couro para o iPhone XS              139
# Capa de silicone para o iPhone XS Max       113
# Capa de couro para o iPhone XS Max          100
# Capa de silicone para o iPhone 11            18
# Capa de couro para o iPhone 11              110
# Capa de silicone para o iPhone 11 Pro        45
# Capa de couro para o iPhone 11 Pro           72

# ----------------- Multiplicacion de un vector -----------------
import numpy as np
vector1 = np.array([2, 3])

# Numero positivo:
vector2 = 2 * vector1
print(vector2)
# Resultado: [4 6]

# Numero negativo: cambia la direccion del vector
vector3 = -1 * vector1
print(vector3)
# Resultado: [-2 -3]

# Graficacion:
plt.figure(figsize=(10.2, 10))
plt.axis([-4, 6, -4, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow3 = plt.arrow(
    0,
    0,
    vector3[0],
    vector3[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(0, 0, 'ro')
plt.legend(
    [arrow1, arrow3, arrow4],
    ['vector1', 'vector3', 'vector4'],
    loc='upper left',
)
plt.grid(True)
plt.show()

# Ejemplo:
import numpy as np
import pandas as pd
quantity_1 = [25, 63, 80, 91, 81, 55, 14, 76, 33, 71]
models = [
    'Silicone case for iPhone 8',
    'Leather case for iPhone 8',
    'Silicone case for iPhone XS',
    'Leather case for iPhone XS',
    'Silicone case for iPhone XS Max',
    'Leather case for iPhone XS Max',
    'Silicone case for iPhone 11',
    'Leather case for iPhone 11',
    'Silicone case for iPhone 11 Pro',
    'Leather case for iPhone 11 Pro',
]
stocks_1 = pd.DataFrame({'Quantity': quantity_1}, index=models)
quantity_2 = [82, 24, 92, 48, 32, 45, 4, 34, 12, 1]
stocks_2 = pd.DataFrame({'Quantity': quantity_2}, index=models)
vector_of_quantity_1 = stocks_1['Quantity'].values
vector_of_quantity_2 = stocks_2['Quantity'].values
vector_of_quantity_united = vector_of_quantity_1 + vector_of_quantity_2
stocks_united = pd.DataFrame(
    {'Quantity': vector_of_quantity_united}, index=models
)
stocks_united['Price'] = [30, 21, 32, 22, 18, 17, 38, 12, 23, 29]
price_united = stocks_united['Price'].values
price_discount_10 = price_united * 0.9
stocks_united['10% discount prise'] = price_discount_10.astype(int)
price_no_discount =  price_discount_10 * 1.1
stocks_united['10% raise price'] = price_no_discount.astype(int)
print(stocks_united)

# ----------------- Valor medio de vectores -----------------
import numpy as np
vector1 = np.array([2, 3])
vector2 = np.array([6, 2])
vector_mean = 0.5 * (vector1 + vector2)
print(vector_mean)
# Resultado: [4.  2.5]

# Graficacion:
plt.figure(figsize=(10, 10))
plt.axis([0, 8.4, -1, 6])
arrow1 = plt.arrow(
    0,
    0,
    vector1[0],
    vector1[1],
    head_width=0.3,
    length_includes_head="True",
    color='b',
)
arrow2 = plt.arrow(
    0,
    0,
    vector2[0],
    vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='g',
)
arrow_sum = plt.arrow(
    0,
    0,
    vector1[0] + vector2[0],
    vector1[1] + vector2[1],
    head_width=0.3,
    length_includes_head="True",
    color='r',
)
arrow_mean = plt.arrow(
    0,
    0,
    vector_mean[0],
    vector_mean[1],
    head_width=0.3,
    length_includes_head="True",
    color='m',
)
plt.plot(vector1[0], vector1[1], 'ro')
plt.plot(vector2[0], vector2[1], 'ro')
plt.plot(vector_mean[0], vector_mean[1], 'ro')
plt.legend(
    [arrow1, arrow2, arrow_sum, arrow_mean],
    ['vector1', 'vector2', 'vector1+vector2', 'vector_mean'],
    loc='upper left',
)
plt.grid(True)
plt.show()

# Ejemplo:
import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

price = ratings['Price'].values  
sum_prices = sum(price)  # suma de todas las valoraciones de precios
average_price_rat = sum(price) / len(price)   
print(average_price_rat)

quality = ratings['Quality'].values
sum_quality = sum(quality)
average_quality_rat =  sum(quality) / len(quality)  
print(average_quality_rat)

average_rat = np.array([average_price_rat, average_quality_rat]) #hallar el vector media

plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])
plt.plot(average_rat[0], average_rat[1], 'mo', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.title('Distribution of ratings and mean value for the whole sample')
plt.show()







import numpy as np
import pandas as pd

ratings_values = [
    [68,18], [81,19], [81,22], [15,75], [75,15], [17,72], 
    [24,75], [21,91], [76, 6], [12,74], [18,83], [20,62], 
    [21,82], [21,79], [84,15], [73,16], [88,25], [78,23], 
    [32, 81], [77, 35]]
ratings = pd.DataFrame(ratings_values, columns=['Price', 'Quality'])

clients_1 = []
clients_2 = []
for client in list(ratings.values):
    if client[0] < 40 and client[1] > 60:
        clients_1.append(client)
    else:
        clients_2.append(client)

average_client_1 = sum(clients_1) / len(clients_1)
print('Valoración media del primer agregador: ', average_client_1)

average_client_2 = sum(clients_2) / len(clients_2)
print('Valoración media del segundo agregador: ', average_client_2)


plt.figure(figsize=(7, 7))
plt.axis([0, 100, 0, 100])

# dibuja la media del grupo 1
# 'b' — azul
plt.plot(average_client_1[0], average_client_1[1], 
         'bo', markersize=15)

# dibuja la media del grupo 2
# 'g' - verde
plt.plot(average_client_2[0], average_client_2[1],
         'go', markersize=15)
plt.plot(price, quality, 'ro')
plt.xlabel('Price')
plt.ylabel('Quality')
plt.grid(True)
plt.title('Distribución de valoraciones y valor medio para cada grupo')
plt.show()

# ----------------- Funciones vectorizadas -----------------
# Funciones que usan vectores

#Division y multiplicacion de dos matrices
import numpy as np
array1 = np.array([2, -4, 6, -8])
array2 = np.array([1, 2, 3, 4])
array_mult = array1 * array2
array_div = array1 / array2
print('Producto de dos matrices: ', array_mult)
print('Cociente de dos matrices: ', array_div)
# Resultado: Producto de dos matrices:  [  2  -8  18 -32]
# Resultado: Cociente de dos matrices:  [ 2. -2.  2. -2.]

# Suma, resta y division de una matriz por un escalar
import numpy as np
array2 = np.array([1, 2, 3, 4])
array2_plus_10 = array2 + 10
array2_minus_10 = array2 - 10
array2_div_10 = array2 / 10
print('Suma: ', array2_plus_10)
print('Resta:', array2_minus_10)
print('Cociente:', array2_div_10)
# Resultado: Suma:  [11 12 13 14]
# Resultado: Resta:  [-9 -8 -7 -6]
# Resultado: Cociente:  [0.1 0.2 0.3 0.4]

# Elevacion al cuadrado de los elementos de una matriz
import numpy as np
numbers_from_0 = np.array([0, 1, 2, 3, 4])
squares = numbers_from_0 ** 2
print(squares)
# Resultado: [ 0  1  4  9 16]

# Funcion que usa vectores
import numpy as np
def min_max_scale(values):
    return (values - min(values)) / (max(values) - min(values))
our_values = np.array([-20, 0, 0.5, 80, -1])
print(min_max_scale(our_values))
# Resultado: [0.    0.2   0.205 1.    0.19 ]

import numpy as np

def logistic_transform(values):
    return 1 / (1 + np.exp(-values))

our_values = np.array([-20, 0, 0.5, 80, -1])
print(logistic_transform(our_values))

# ----------------- Aplicar la vectorización a las métricas de evaluación -----------------
import numpy as np
target = np.array([0.9, 1.2, 1.4, 1.5, 1.9, 2.0])
predictions = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

# ----------------- Funciones aplicadas a vectores -----------------
# Missma funcion que abajo, una usa suma y otra media
def mse1(target, predictions):
    n = target.size
    return ((target - predictions) ** 2).sum() / n

def mse2(target, predictions):
    return ((target - predictions) ** 2).mean()

def mae(target, predictions):
    return np.abs((target - predictions)).mean()

def rmse(target, predictions):
    return (((target - predictions) ** 2).mean()) ** 0.5

# ----------------- producto punto o producto escalar -----------------
# ¿Qué es el producto punto?
# Es una operación entre dos vectores que produce un número único (escalar)
# Se multiplican los elementos correspondientes y se suman todos los resultados.
precios = [10, 20, 30]     
cantidades = [2, 3, 1]    
# Producto punto: (10 × 2) + (20 × 3) + (30 × 1) = 20 + 60 + 30 = 110


import numpy as np
no_of_hours = np.array([160, 150, 80, 95, 60, 40])
hourly_rate = np.array([30, 35, 40, 40, 45, 50])

print(sum(no_of_hours * hourly_rate)) # Importe total, número que es un escalar 
print(np.dot(no_of_hours , hourly_rate )) # Mismo uso
print(no_of_hours @ hourly_rate ) # Operador de multiplicación de matrices

# Resultado:  21750

# Que es un escalar?
Un escalar es un solo valor numérico que representa una cantidad o magnitud.
los escalares son unidimensionales.
Un escalar es simplemente un número único. No tiene dirección, solo magnitud (tamaño)

# Escalar:
temperatura = 25  # Un solo número

# Vector:
precios = [10, 20, 30]  # Una lista de números

# Acerca de los vectores
los vectores deben tener el mismo tamaño
La aplicación del producto escalar es esencial para los algoritmos de machine learning basados en la distancia.

import numpy as np
import pandas as pd
store1_price = [
    209.9,
    119.9,
    53.9,
    31.9,
    19.9,
    109.9,
    59.99,
    22.9,
    81.11,
    32.9,
]
store1_quantity = [19, 11, 8, 15, 23, 7, 14, 9, 10, 4]
store2_price = [209.9, 124.9, 42.9, 27.9, 23.9, 109.9, 49.9, 24.9, 89.9, 32.9]
store2_quantity = [10, 16, 20, 9, 18, 12, 10, 11, 18, 22]
models = [
    'Apple AirPods Pro',
    'Apple AirPods MV7N2RU/A',
    'JBL Tune 120TWS',
    'JBL TUNE 500BT',
    'JBL JR300BT',
    'Huawei Freebuds 3',
    'Philips TWS SHB2505',
    'Sony WH-CH500',
    'Sony WF-SP700N',
    'Sony WI-XB400',
]
items1 = pd.DataFrame(
    {'Price': store1_price, 'Quantity': store1_quantity}, index=models
)
items2 = pd.DataFrame(
    {'Price': store2_price, 'Quantity': store2_quantity}, index=models
)
print('Store 1\n', items1, '\n')
print('Store 2\n', items2)
items1_price = items1['Price'].values 
items1_quantity = items1['Quantity'].values 
items2_price = items2['Price'].values 
items2_quantity = items2['Quantity'].values 
items1_value = np.dot(items1_price, items1_quantity)
items2_value = np.dot(items2_price, items2_quantity)
total_value = items1_value + items2_value 
print('Valor total de los artículos de las dos tiendas:', total_value, 'USD')

# ----------------- Distancia euclidiana -----------------
# Es siempre la distancia más corta entre dos puntos en un plano.


import numpy as np
a = np.array([5, 6])

print(((a[0] ** 2) + (a[1] ** 2)) ** 0.5) # mediante el teorema de Pitágoras
print(np.dot(a, a) ** 0.5) # mediante el producto punto 
# Resultado:  7.810249675906654

podemos pensar en la diferencia entre los dos vectores como en un nuevo vector (b−a)

import numpy as np
a = np.array([5, 6])
b = np.array([1, 3])
d = np.dot(b - a, b - a) ** 0.5
print('La distancia entre a y b es de', d)

import numpy as np
from scipy.spatial import distance
a = np.array([5, 6])
b = np.array([1, 3])
d = distance.euclidean(a, b)
print('La distancia entre a y b es de', d)

import numpy as np
import pandas as pd
from scipy.spatial import distance
x_axis = np.array([0., 0.18078584, 9.32526599, 17.09628721,
                      4.69820241, 11.57529305, 11.31769349, 14.63378951])
y_axis  = np.array([0.0, 7.03050245, 9.06193657, 0.1718145,
                      5.1383203, 0.11069032, 3.27703365, 5.36870287])
deliveries = np.array([5, 7, 4, 3, 5, 2, 1, 1])
town = [
    'Willowford',
    'Otter Creek',
    'Springfield',
    'Arlingport',
    'Spadewood',
    'Goldvale',
    'Bison Flats',
    'Bison Hills',
]
data = pd.DataFrame(
    {
        'x_coordinates_km': x_axis,
        'y_coordinates_km': y_axis,
        'Deliveries': deliveries,
    },
    index=town,
)
vectors = data[['x_coordinates_km', 'y_coordinates_km']].values
distances = []
for town_from in range(len(town)):
    row = []
    for town_to in range(len(town)):
        value = distance.euclidean(vectors[town_from], vectors[town_to])
        row.append(value)
    distances.append(row)
deliveries_in_week = []
for i in range(len(town)):
    value = 2 * np.dot(np.array(distances[i]), deliveries)
    deliveries_in_week.append(value)
deliveries_in_week_df = pd.DataFrame(
    {'Distance': deliveries_in_week}, index=town
)
print(deliveries_in_week_df)
print()
print('Localidad del almacén:', deliveries_in_week_df['Distance'].idxmin())

# ----------------- Distancia Manhattan o distancia entre manzanas -----------------
Es la busqueda de un camino que no es linea recta, sino que sigue una cuadrícula ortogonal (como las calles de una ciudad).
Es la suma de las diferencias absolutas de las coordenadas de todos los puntos

import numpy as np
def manhattan_distance(first, second):
    return np.abs(first - second).sum()
first = np.array([3, 11])
second = np.array([1, 6])
print(manhattan_distance(first, second))

import numpy as np
import pandas as pd
from scipy.spatial import distance

# Crea DataFrames para las avenidas y calles
avenues_df = pd.DataFrame([0, 153, 307, 524], index=['Park', 'Lexington', '3rd', '2nd'], columns=['Distance'])
 
						  
streets_df = pd.DataFrame([0, 81, 159, 240, 324], index=['76', '75', '74', '73', '72'], columns=['Distance'])
 

# Datos de direcciones y taxis
address = ['Lexington', '74']
taxis = [
    ['Park', '72'],
    ['2nd', '75'],
    ['3rd', '76'],
]

# Convierte dirección en vector
address_vector = np.array([avenues_df.loc[address[0], 'Distance'], streets_df.loc[address[1], 'Distance']])

# Calcula distancias
taxi_distances = []
for taxi in taxis:
    taxi_vector = np.array([avenues_df.loc[taxi[0], 'Distance'], streets_df.loc[taxi[1], 'Distance']])
    taxi_distances.append(distance.cityblock(address_vector, taxi_vector))

# Encuentra el índice del taxi más cercano
index = np.argmin(taxi_distances)
print(taxis[index])

# -----------------  -----------------
tener los datos representados en un vector multidimensional, permite encontrar información  utilizando la distancia euclidiana como métrica de distancia.
Esta métrica de distancia puede ayudarnos a identificar el grado de relación entre diferentes observaciones.

Incluso cuando el número de coordenadas es superior a dos, podemos utilizar (para calcular distancias en el espacio multidimensional):
    distance.euclidean()
    distance.cityblock() 

# distancia multidimensional, mas de 2 coordenadas, con la distancia euclidiana
import numpy as np 
from scipy.spatial import distance
a = np.array([4, 2, 3, 0, 5])
b = np.array([1, 0, 3, 2, 6])
d = np.dot(b - a, b - a)**0.5 # cálculo manual
print(d)
print()
e = distance.euclidean(a, b) # cálculo con la función euclidiana
print(e)

# distancia multidimensional, mas de 2 coordenadas, con la distancia Manhattan
import numpy as np
from scipy.spatial import distance
a = np.array([4, 2, 3, 0, 5])
b = np.array([1, 0, 3, 2, 6])
d = np.abs(b - a).sum()
print(d)
print()
e = distance.cityblock(a, b)
print(e)

import pandas as pd
from scipy.spatial import distance

columns = [
    'dormitorios',
    'superficie total',
    'cocina',
    'superficie habitable',
    'planta',
    'número de plantas',
]
realty = [
    [1, 38.5, 6.9, 18.9, 3, 5],
    [1, 38.0, 8.5, 19.2, 9, 17],
    [1, 34.7, 10.3, 19.8, 1, 9],
    [1, 45.9, 11.1, 17.5, 11, 23],
    [1, 42.4, 10.0, 19.9, 6, 14],
    [1, 46.0, 10.2, 20.5, 3, 12],
    [2, 77.7, 13.2, 39.3, 3, 17],
    [2, 69.8, 11.1, 31.4, 12, 23],
    [2, 78.2, 19.4, 33.2, 4, 9],
    [2, 55.5, 7.8, 29.6, 1, 25],
    [2, 74.3, 16.0, 34.2, 14, 17],
    [2, 78.3, 12.3, 42.6, 23, 23],
    [2, 74.0, 18.1, 49.0, 8, 9],
    [2, 91.4, 20.1, 60.4, 2, 10],
    [3, 85.0, 17.8, 56.1, 14, 14],
    [3, 79.8, 9.8, 44.8, 9, 10],
    [3, 72.0, 10.2, 37.3, 7, 9],
    [3, 95.3, 11.0, 51.5, 15, 23],
    [3, 69.3, 8.5, 39.3, 4, 9],
    [3, 89.8, 11.2, 58.2, 24, 25],
]

df_realty = pd.DataFrame(realty, columns=columns)

vector_first = df_realty.loc[3].values 
vector_second = df_realty.loc[11].values 

print('Distancia euclidiana:', distance.euclidean(vector_first, vector_second))
print('Distancia Manhattan:', distance.cityblock(vector_first, vector_second))

# índice del apartamento preferido
preference_index = 12
preference_vector = df_realty.loc[preference_index].values

distances = []
for i in range(df_realty.shape[0]):
    other_vector = df_realty.loc[i].values
    distances.append(distance.euclidean(preference_vector, other_vector))

best_index = np.array(distances).argsort()[1] # argsort() devuelve los índices en orden
# [1] porque el primer valor es el index 12

print('Índice del apartamento más similar:', best_index)

# ----------------- algoritmo K-Nearest Neighbors (k-NN) -----------------
utilizado a la hora de realizar tareas de clasificación y regresión
basado en la distancia
Utilizar la observación más cercana para clasificar nuestra nueva observación

Paso 1: Seleccionamos el número (k) de vecinos de referencia.
Paso 2: Calculamos la distancia entre la nueva observación y los demás puntos. La distancia euclidiana por defecto, pero podemos utilizar otras
Paso 3: Ordenamos las distancias de la más cercana a la más lejana y consideramos los vecinos k más cercanos.
Paso 4: Finalmente, clasificamos nuestra nueva observación en la misma clase a la que pertenece la mayoría de los vecinos.

import numpy as np
import pandas as pd
from scipy.spatial import distance

# Algoritmo que puede predecir cuántos dormitorios tiene un apartamento basándose en otras características
columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 71.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 9.5, 42.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


def nearest_neighbor_predict(train_features, train_target, new_features):
    distances = []
    for i in range(train_features.shape[0]):
        vector = train_features.loc[i].values
        distances.append(distance.euclidean(new_features, vector))
    best_index = np.array(distances).argmin()
    print(distances)
    return train_target.loc[best_index]


train_features = df_train.drop('bedrooms', axis=1)
train_target = df_train['bedrooms']
new_apartment = np.array([72, 14, 39, 8, 16])
prediction = nearest_neighbor_predict(
    train_features, train_target, new_apartment
)
print(prediction)

# ----------------- Creacion de una clase personalizada -----------------
import pandas as pd
from scipy.spatial import distance

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
    'price',
]

df_train = pd.DataFrame(
    [
        [1, 38.5, 6.9, 18.9, 3, 5, 4200000],
        [1, 38.0, 8.5, 19.2, 9, 17, 3500000],
        [1, 34.7, 10.3, 19.8, 1, 9, 5100000],
        [1, 45.9, 11.1, 17.5, 11, 23, 6300000],
        [1, 42.4, 10.0, 19.9, 6, 14, 5900000],
        [1, 46.0, 10.2, 20.5, 3, 12, 8100000],
        [2, 77.7, 13.2, 39.3, 3, 17, 7400000],
        [2, 69.8, 11.1, 31.4, 12, 23, 7200000],
        [2, 78.2, 19.4, 33.2, 4, 9, 6800000],
        [2, 55.5, 7.8, 29.6, 1, 25, 9300000],
        [2, 74.3, 16.0, 34.2, 14, 17, 10600000],
        [2, 78.3, 12.3, 42.6, 23, 23, 8500000],
        [2, 74.0, 18.1, 49.0, 8, 9, 6000000],
        [2, 91.4, 20.1, 60.4, 2, 10, 7200000],
        [3, 85.0, 17.8, 56.1, 14, 14, 12500000],
        [3, 79.8, 9.8, 44.8, 9, 10, 13200000],
        [3, 72.0, 10.2, 37.3, 7, 9, 15100000],
        [3, 95.3, 11.0, 51.5, 15, 23, 9800000],
        [3, 69.3, 8.5, 39.3, 4, 9, 11400000],
        [3, 89.8, 11.2, 58.2, 24, 25, 16300000],
    ],
    columns=columns,
)

train_features = df_train.drop('price', axis=1)
train_target = df_train['price']

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7, 3900000],
        [2, 71.7, 12.2, 34.3, 5, 21, 7100000],
        [3, 88.0, 18.1, 58.2, 17, 17, 12100000],
    ],
    columns=columns,
)

test_features = df_test.drop('price', axis=1)


class ConstantRegression:
    def fit(self, train_features, train_target):
        self.mean = train_target.mean()

    def predict(self, new_features):
        answer = pd.Series(self.mean, index=new_features.index)
        return answer


model = ConstantRegression()
model.fit(train_features, train_target)
test_predictions = model.predict(test_features)
print(test_predictions)


import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = [
    "bedrooms",
    "total area",
    "kitchen",
    "living area",
    "floor",
    "total floors",
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 91.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 8.5, 39.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


train_features = df_train.drop("bedrooms", axis=1)
train_target = df_train["bedrooms"]

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7],
        [2, 71.7, 12.2, 34.3, 5, 21],
        [3, 88.0, 18.1, 58.2, 17, 17],
    ],
    columns=columns,
)

test_features = df_test.drop("bedrooms", axis=1)

class NearestNeighborClassifier:
    def fit(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target

model = NearestNeighborClassifier()
model.fit(train_features, train_target)
print(model.train_features.head())
print(model.train_target.head())


import numpy as np
import pandas as pd
from scipy.spatial import distance

columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 91.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 8.5, 39.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


train_features = df_train.drop('bedrooms', axis=1)
train_target = df_train['bedrooms']

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7],
        [2, 71.7, 12.2, 34.3, 5, 21],
        [3, 88.0, 18.1, 58.2, 17, 17],
    ],
    columns=columns,
)

test_features = df_test.drop('bedrooms', axis=1)


def nearest_neighbor_predict(train_features, train_target, new_features):
    distances = []
    for i in range(train_features.shape[0]):
        vector = train_features.loc[i].values
        distances.append(distance.euclidean(new_features, vector))
    best_index = np.array(distances).argmin()
    return train_target.loc[best_index]


class NearestNeighborClassifier:
    def fit(self, train_features, train_target):
        self.train_features = train_features
        self.train_target = train_target

    def predict(self, new_features):
        values = []
        for i in range(new_features.shape[0]):
            vector = new_features.loc[i]
            values.append(
                nearest_neighbor_predict(
                    self.train_features, self.train_target, vector
                )
            )
        return pd.Series(values)


model = NearestNeighborClassifier()
model.fit(train_features, train_target)
new_predictions = model.predict(test_features)
print(new_predictions)



















































































































































































































# ----------------- Creacion de matrices -----------------
Es una estructura de datos bidimensional que almacena datos en filas y columnas(Tabla).

# Matriz 3x3 con valores definidos como argumento
import numpy as np
matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix)

# Matriz de 2×3 con variables como argumento
import numpy as np
vector1 = np.array([1, 2, 3])
vector2 = np.array([-1, -2, -3])
list_of_vectors = [vector1, vector2]
matrix_from_vectors = np.array(list_of_vectors)
print(matrix_from_vectors)

# Dataframe a matriz
import pandas as pd
df = pd.DataFrame({'a': [120, 60, 75], 'b': [42, 50, 90]})
matrix = df.values
print(df)
print()
print(matrix)

# Resultado:
#      a   b
# 0  120  42
# 1   60  50
# 2   75  90

# [[120  42]
# [ 60  50]
# [ 75  90]]

# Saber el tamaño de una matriz
import numpy as np
A = np.array([
    [1, 2, 3], 
    [2, 3, 4]])
print('Tamaño:', A.shape) 

# Resultado:
# Tamaño: (2, 3) 

# Acceder a un elemento de una matriz
import numpy as np
A = np.array([
    [1, 2, 3], 
    [2, 3, 4]])
print('A[1, 2]:', A[1, 2])
# Resultado:
# A[1, 2]: 4

# Obtener fila y columna
import numpy as np
matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]])
print('Primera fila:', matrix[0, :])
print('Tercera columna:', matrix[:, 2])
# Resultado:
# Primera fila: [1 2 3]
# Tercera columna: [ 3  6  9 12]

# ----------------- Operaciones con elementos de matrices -----------------
# Mismas operaciones que con vectores
# Las matrices en las operaciones deben ser del mismo tamaño

# Suma de matrices
import numpy as np
matrix1 = np.array([
    [1, 2], 
    [3, 4]])
matrix2 = np.array([
    [5, 6], 
    [7, 8]])
print(matrix1 + matrix2) 
# Resultado:
# [[ 6  8]
#  [10 12]]

# Resta de matrices
import numpy as np
matrix1 = np.array([
    [5, 10], 
    [1, 20]])
matrix2 = np.array([
    [1, 3], 
    [3, 10]])
print(matrix1 - matrix2)
# Resultado:
# [[ 4  7]
#  [-2 10]]

# Multiplicación de matrices
import numpy as np
matrix1 = np.array([
    [1, 2], 
    [3, 4]])
matrix2 = np.array([
    [5, 6], 
    [7, 8]])
print(np.multiply(matrix1, matrix2)) 
# Resultado:
# [[ 5 12]
#  [21 32]]

# Operaciones con un escalar
import numpy as np
matrix = np.array([
    [1, 2], 
    [3, 4]])
print(matrix * 2)
# Resultado:
# [[2 4]
#  [6 8]]
print(matrix + 2)
# Resultado:
# [[3 4]
#  [5 6]]
print(matrix - 2)
# Resultado:
# [[-1  0]
#  [ 1  2]]

# Ejercicio de la leccion
import numpy as np
import pandas as pd
services = ['Minutos', 'Mensajes', 'Megabytes']
monday = np.array([
    [10, 2, 72],
    [3, 5, 111],
    [15, 3, 50],
    [27, 0, 76],
    [7, 1, 85]])
tuesday = np.array([
    [33, 0, 108],
    [21, 5, 70],
    [15, 2, 15],
    [29, 6, 34],
    [2, 1, 146]])
wednesday = np.array([
    [16, 0, 20],
    [23, 5, 34],
    [5, 0, 159],
    [35, 1, 74],
    [5, 0, 15]])
thursday = np.array([
    [25, 1, 53],
    [15, 0, 26],
    [10, 0, 73],
    [18, 1, 24],
    [2, 2, 24]])
friday = np.array([
    [32, 0, 33],
    [4, 0, 135],
    [2, 2, 21],
    [18, 5, 56],
    [27, 2, 21]])
saturday = np.array([
    [28, 0, 123],
    [16, 5, 165],
    [10, 1, 12],
    [42, 4, 80],
    [18, 2, 20]])
sunday = np.array([
    [30, 4, 243],
    [18, 2, 23],
    [12, 2, 18],
    [23, 0, 65],
    [34, 0, 90]])
weekly = monday + sunday + saturday + friday + thursday + tuesday + wednesday
print('En una semana')
print(pd.DataFrame(weekly, columns=services))
print()
forecast = (weekly * 30.4) / 7
print('Pronóstico para un mes')
print(pd.DataFrame(forecast, dtype=int, columns=services))

# ----------------- Multiplicar una matriz por un vector -----------------
import numpy as np
matrix = np.array([
    [1, 2, 3], 
    [4, 5, 6]])
vector = np.array([7, 8, 9])
# función independiente - toma dos argumentos
print(np.dot(matrix, vector))
# método de la matriz - solo necesita un argumento
print(matrix.dot(vector))
# Mismo resultado:
# [ 50 122]

# Ejercicio de la leccion
import numpy as np
import pandas as pd
services = ['Minutos', 'Mensajes', 'Megabytes']
packs_names = ['"Al volante"', '"En el metro"']
packs = np.array([[20, 5], [2, 5], [500, 1000]])
clients_packs = np.array([[1, 2], [2, 3], [4, 1], [2, 3], [5, 0]])
print("Paquete del cliente")
print(pd.DataFrame(clients_packs[1], index=packs_names, columns=['']))
print()
client_vector = clients_packs[1, :]
client_services = np.dot(packs, client_vector)
print('Minutos', 'Mensajes', 'Megabytes')
print(client_services)

# ----------------- Transposicion de matriz  -----------------
# Transpuesta: intercambiar las columnas y las filas de una matriz
# La "volteamos" sobre la diagonal principal de la matriz(esquina superior izquierda hasta la esquina inferior derecha)
# 𝑚×𝑛 a n×m

import numpy as np
matrix = np.array([
    [1, 2], 
    [4, -4], 
    [0, 17]])
print('Matriz original')
print(matrix)
print('Matriz transpuesta')
print(matrix.T)

# Resultado:
# Matriz original
# [[ 1  2]
#  [ 4 -4]
#  [ 0 17]]
# Matriz transpuesta
# [[ 1  4  0]
#  [ 2 -4 17]]

# Intentar multiplicar la matriz original por un vector de diferente tamaño
vector = [2, 1, -1]
print(np.dot(matrix, vector))
# Resultado:
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# ... in <module>
#       1 vector = [2, 1, -1]
# ----> 2 print(np.dot(matrix, vector))
# ValueError: shapes (3,2) and (3,) not aligned: 2 (dim 1) != 3 (dim 0)

# Tenemos un error: las dimensiones de la matriz (3, 2) y el vector (3,) no están alineados

print(np.dot(matrix.T, vector))
# Resultado:
# [  6 -17]

# Ejercicio de la leccion
import numpy  as np
import pandas as pd
materials_names = ['Tabla', 'Tubos', 'Tornillos']
manufacture = np.array([
    [0.2, 1.2, 8],
    [0.5, 0.8, 6],
    [0.8, 1.6, 8]])
furniture = [60, 4, 16]
materials = np.dot(manufacture.T, furniture)
print(pd.DataFrame([materials], index=[''], columns=materials_names))

# ----------------- Multiplicación de matrices -----------------
Durante la multiplicación de matrices, se construye una tercera matriz usando dos matrices. 
Solo es posible si el número de columnas de la primera matriz (𝑚×𝑛) es igual al número de filas de la segunda matriz (𝑛×𝑝).

# Formas de multiplicacion
import numpy as np
    
A = np.array([
    [1, 2, 3], 
    [-1, -2, -3]])
B = np.array([
    [1, 0], 
    [0, 1],
    [1, 1]])
print(np.dot(A,B)) 
print(A.dot(B)) 
print(A @ B)
# Resultado:
# [[ 4  5]
#  [-4 -5]]

El resultado de la multiplicación de matrices depende del orden de los multiplicadores
print(B @ A)
# Resultado:
# [[ 1  2  3]
#  [-1 -2 -3]
#  [ 0  0  0]]

Multiplicar la matriz por sí misma solo es posible si es cuadrada
square_matrix = np.array([
    [1, 2, 3], 
    [-1, -2, -3],
    [0, 0, 0]])
print(square_matrix @ square_matrix)
# Resultado:
# [[-1 -2 -3]
#  [ 1  2  3]
#  [ 0  0  0]]

Siempre que el número de filas en la matriz bookings sea igual al número de columnas en la matriz costs, la multiplicación funciona. 
import numpy as np
import pandas as pd
months_1 = ['February', 'March']
months_2 = ['June', 'July', 'August']
costs = np.array([
    [700, 50] 
])
bookings_1 = np.array([
    [3, 4], # número de reservas de suites de luna de miel en febrero y marzo
    [17, 26] # número de reservas de habitaciones regulares en febrero y marzo
])
bookings_2 = np.array([
    [8, 9, 11], # número de reservas de suites de luna de miel en junio, julio y agosto
    [21, 26, 30] # número de reservas de habitaciones regulares en junio, julio y agosto
])
print(pd.DataFrame((costs @ bookings_1), columns=months_1))
print('----------------------')
print(pd.DataFrame((costs @ bookings_2), columns=months_2))
# Resultado:
#    February  March
# 0      2950   4100
# ----------------------
#    June  July  August
# 0  6650  7600    9200

# Ejercicio de la leccion:
import numpy as np
import pandas as pd
services = ['Minutos', 'Mensajes', 'Megabytes']
packs_names = ['"Detrás del volante', '"En el metro"']
packs = np.array([
    [20, 5],
    [2, 5],
    [500, 1000]])
clients_packs = np.array([
    [1, 2],
    [2, 3],
    [4, 1],
    [2, 3],
    [5, 0]])
print('Paquetes')
print(pd.DataFrame(clients_packs, columns=packs_names))
print()
clients_services = (packs @ clients_packs.T).T
print('Minutos, Mensajes y Megabytes')
print(pd.DataFrame(clients_services, columns=services))
print()

import numpy  as np
import pandas as pd
materials_names = ['Tabla', 'Tubos', 'Tornillos']
venues_names = ['Cafetería', 'Comedor', 'Restaurante']
manufacture = np.array([
    [0.2, 1.2, 8],    # materiales para silla
    [0.5, 0.8, 6],    # materiales para banco
    [0.8, 1.6, 8]])   # materiales para mesa
furniture = np.array([
    [12, 0, 3],   # Pedido de cafetería (silla, banca, mesa)
    [40, 2, 10],  # Pedido de comedor
    [60, 6, 18]]) # Pedido de restaurante
venues_materials = furniture @ manufacture
print('Por el establecimiento')
print(pd.DataFrame(venues_materials, index=venues_names, columns=materials_names))
print()
venues = [18, 12, 7]
total_materials = np.dot(venues, venues_materials)
print('Total')
print(pd.DataFrame([total_materials], index=[''], columns=materials_names))

# ----------------- Modelo de regresión lineal -----------------
La regresión lineal se usa para hacer una predicción objetivo basada en un conjunto de características.

La predicción de un modelo (a) se calcula entonces multiplicando la matriz de características X por el vector de peso (w),
más el valor del sesgo de predicción (w0).

Predicción = (X · w) + w₀
Donde:
- X = vector de características
- w = vector de pesos (uno para cada característica)
- w₀ = sesgo (bias/intercept)

El sesgo es como el punto de partida de tu predicción. Sin él, tu línea de regresión siempre pasaría por el origen (0,0), lo cual rara vez es realista.

Cambiar solo el parámetro w0 desplazaría toda la línea hacia arriba o hacia abajo en el gráfico(Direccion)
Cuanto mayor sea la distancia de la línea a los datos, mayor será el valor de la métrica ECM.

El parámetro w determina la pendiente de la línea de regresión.
El parámetro w0 sube o baja la línea recta.

Aumentar/Disminuir w₀ (Esto hara que el valor de las predicciones bajen o suban) hara que la métrica ECM disminuirá.

# ----------------- Objetivo de entrenamiento -----------------
Una función objetivo es una función que debe minimizarse o maximizarse para resolver un problema de optimización. 
Esta función toma datos y parámetros del modelo como argumentos, los evalúa y devuelve un número.
Nuestro propósito es encontrar los valores de estos parámetros que minimizan o maximizan el resultado.

Cuando trabajamos con regresión lineal, nuestra función objetivo es una función de pérdida que mide el rendimiento de predicción del modelo. 
Entonces, la meta de la función objetivo es minimizar el error cuadrático medio (ECM)

Nuestro objetivo de entrenamiento es encontrar los valores ideales para estos parámetros (w y w₀) que minimizan la función de pérdida (ECM) en nuestro conjunto de datos de entrenamiento.

# ----------------- Matriz inversa -----------------
Una matriz identidad (también conocida como matriz unitaria) es una matriz cuadrada con unos a lo largo de la diagonal principal y ceros en el resto.
    1 0 0
E = 0 1 0   
    0 0 1

Si cualquier matriz A se multiplica por una matriz identidad (o viceversa), el resultado será la matriz 
A⋅E=E⋅A=A

La matriz inversa (denotada por A⁻¹) para una matriz cuadrada A es la matriz cuyo producto con A es igual a la matriz identidad.
No todas las matrices tienen inversa, y las que la tienen se llaman matrices invertibles.
A⋅A⁻¹ = A⁻¹ ⋅A=E

Solo las matrices cuadradas pueden ser invertibles. Sin embargo, no toda matriz cuadrada tiene una inversa.

# Crear una matriz inversa
import numpy as np
np.random.randint(lower_limit,upper_limit,(rows,columns)).

# Encontrar la matriz inversa
np.linalg.inv()

# ----------------- Entrenamiento de una regresión lineal -----------------
Pasos para la funcion de perdida ECM:
    1. La matriz de características transpuesta se multiplica por sí misma.
    2. Se calcula la matriz inversa a ese resultado.
    3. La matriz inversa se multiplica por la matriz de características transpuesta.
    4. El resultado se multiplica por el vector de los valores objetivo.

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']
data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)
features = data.drop('price', axis=1)
target = data['price']
class LinearRegression:
    def fit(self, train_features, train_target):
        self.w = None
        self.w0 = None
    def predict(self, test_features):
        return np.zeros(test_features.shape[0])
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']
data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)
features = data.drop('price', axis=1)
target = data['price']
class LinearRegression:
    def fit(self, train_features, train_target):
        self.w = np.zeros(train_features.shape[1])
        self.w0 = train_target.mean()
    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0 #el valor solo es la media, ya que los pesos son 0 y hace que features no tenga valor
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
columns = ['bedrooms', 'total area', 'kitchen', 'living area', 'floor', 'total floors', 'price']
data = pd.DataFrame([
    [1, 38.5, 6.9, 18.9, 3, 5, 84000],
    [1, 38.0, 8.5, 19.2, 9, 17, 70000],
    [1, 34.7, 10.3, 19.8, 1, 9, 102000],
    [1, 45.9, 11.1, 17.5, 11, 23, 126000],
    [1, 42.4, 10.0, 19.9, 6, 14, 118000],
    [1, 46.0, 10.2, 20.5, 3, 12, 162000],
    [2, 77.7, 13.2, 39.3, 3, 17, 148000],
    [2, 69.8, 11.1, 31.4, 12, 23, 144000],
    [2, 78.2, 19.4, 33.2, 4, 9, 136000],
    [2, 55.5, 7.8, 29.6, 1, 25, 186000],
    [2, 74.3, 16.0, 34.2, 14, 17, 212000],
    [2, 78.3, 12.3, 42.6, 23, 23, 170000],
    [2, 74.0, 18.1, 49.0, 8, 9, 120000],
    [2, 91.4, 20.1, 60.4, 2, 10, 144000],
    [3, 85.0, 17.8, 56.1, 14, 14, 250000],
    [3, 79.8, 9.8, 44.8, 9, 10, 264000],
    [3, 72.0, 10.2, 37.3, 7, 9, 302000],
    [3, 95.3, 11.0, 51.5, 15, 23, 196000],
    [3, 69.3, 8.5, 39.3, 4, 9, 228000],
    [3, 89.8, 11.2, 58.2, 24, 25, 326000],
], columns=columns)
features = data.drop('price', axis=1)
target = data['price']
class LinearRegression:
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w =  np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = w[1:]
        self.w0 = w[0]
    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0
model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(r2_score(target, predictions))

# ----------------- ¿Qué es enmascaramiento de datos u ofuscación de datos? -----------------
data masking o data obfuscation es el conjunto de técnicas usadas para proteger información sensible, alterando los datos de forma controlada para que no se pueda identificar a personas u organizaciones reales, pero sin perder utilidad analítica.
los datos “parecen reales”, se comportan como reales, pero no revelan información confidencial.

Enmascaramiento: oculta o transforma datos manteniendo estructura y estadísticas

Ofuscación: distorsiona datos para hacerlos difíciles de interpretar, incluso si se ven















































































































































































































































































































































































































































































































































