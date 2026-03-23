#---------------------------- análisis de clústeres ----------------------------
# Durante el aprendizaje supervisado, el modelo encuentra la relación entre las características y la característica objetivo con el fin de hacer predicciones mediante nuevas observaciones.

# En el aprendizaje no supervisado no hay característica objetivo
# el modelo busca relaciones entre las observaciones 

# El análisis de clústeres (clustering) es la tarea de combinar observaciones similares en grupos o clústeres.

# La mayoría de los métodos de clustering implican determinar la similitud o diferencia de las observaciones con base en la distancia entre ellas.
# Cuanto más lejos estén las observaciones entre sí, menos similares serán, y viceversa.

# en  la clasificación, las clases y las observaciones se conocen por adelantado
# En el análisis de clústeres, no se especifican

# se definen de diversas maneras dependiendo de cuáles son las observaciones que se consideran similares

# Digamos que quieres organizar la biblioteca de tu casa. 
# Escoge cuatro libros de diferentes géneros y determina estantes distintos para cada uno. 
# Todo lo que queda por hacer es organizar los géneros de los otros libros y colocarlos en los estantes correspondientes.

# Encontrando una forma de agrupar los usuarios por similitud y dividirlos en grupos.
# Lo único que tienes que hacer es seleccionar la característica que determinará la similitud entre usuarios.

# ¿Cuáles podrian ser características que pueden determinar la similitud entre usuarios? 
# - Tiempo promedio de sesión.
# - El número de meses desde la primera compra.
# - El pago más grande.

#---------------------------- ¿Qué es un cluster? ----------------------------
# Un cluster es un grupo de observaciones (datos) que son similares entre sí.

# Analogía simple:
# Imagina que tienes una caja llena de frutas mezcladas:
#     - Cluster 1: Todas las manzanas (rojas, redondas, dulces)
#     - Cluster 2: Todas las naranjas (naranjas, redondas, cítricas)
#     - Cluster 3: Todos los plátanos (amarillos, alargados, dulces)

# El clustering agrupa datos sin etiquetas previas.

# Por ejemplo, segmentación de usuarios:
#     - Cluster básico: Usuarios con pocas compras, sesiones cortas
#     - Cluster avanzado: Usuarios moderadamente activos
#     - Cluster premium: Usuarios con compras grandes, muy activos

# Diferencia clave:
# Clasificación: Ya sabes las categorías (spam/no spam)
# Clustering: Descubres los grupos automáticamente

# ¿Cómo se forman?
# Los algoritmos como k-means agrupan datos por distancia:
# Datos cercanos = similares = mismo cluster
# Datos lejanos = diferentes = clusters distintos

# Ejemplos prácticos:
# Agrupar clientes por comportamiento de compra
# Segmentar imágenes por color y textura

#---------------------------- clustering de k-means ----------------------------
# el algoritmo de clustering más popular

# El concepto clave de este algoritmo es el centroide, o el centro de un clúster. 
# El grado de cercanía a un centro en particular depende de a qué clúster pertenece la observación. 
# Cada clúster tiene su propio centroide, que se calcula como la media aritmética de las observaciones agrupadas.

# Límites técnicos:
# - Mínimo: 1 cluster (no tiene sentido)
# - Máximo teórico: Tantos como observaciones tengas
# - Máximo práctico: Depende de tus datos

# 2 clusters: Usuarios básicos vs premium
# 3 clusters: Básicos, intermedios, premium 
# 5 clusters: Segmentación más detallada
# 10+ clusters: Micro-segmentación muy específica

# Pocos clusters: Grupos amplios, fáciles de interpretar
# Muchos clusters: Grupos específicos, difíciles de manejar

# K-means segmenta las observaciones paso a paso, así que es un algoritmo iterativo.
# 1. El algoritmo asigna al azar un número de clúster a cada observación, de 1 a k.
# 2. El algoritmo repite un proceso de iteración de dos pasos hasta que los clústeres de observación dejan de cambiar:
#     - Se calcula el centroide de cada clúster.
#     - A cada observación se le asigna el número de un nuevo clúster cuyo centroide está más cerca de la observación.

# Otra condición para detener el trabajo del algoritmo es el parámetro para el máximo número de iteraciones (max_iter). 

# Al lanzarlo, las observaciones tienen un color al azar. 
# Los colores denotan números de clústeres. 
# Después de una iteración del algoritmo, las observaciones de un color se combinan y los clústeres adquieren límites claros.

# el valor promedio de los vectores xʲ es el producto de la suma de todos los vectores y 1/N, donde N es el número de vectores. 

# la primera coordenada del vector resultante es la suma de las primeras coordenadas de todos los vectores
# la segunda coordenada es la suma de las segundas
# así hasta n (dimensiones de vector)

# Básico: tiempo promedio de sesión (timespent): 50 minutos; pago promedio (purchase): $20 000; 5 meses desde el registro (months).
# Avanzado: tiempo promedio de sesión: 20 minutos; pago promedio: $30 000; 10 meses desde el registro.
# Premium: tiempo promedio de sesión: 20 minutos; pago promedio: $80 000; 8 meses desde el registro.

# Necesitamos decirle al algoritmo dónde buscar los clústeres. 
# Pasaremos los valores de estas características como entrada a k-means para configurar los centroides iniciales

#---------------------------- ¿Qué hace RuntimeWarning? ----------------------------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Silencia las advertencias de tipo RuntimeWarning durante la ejecución.

# ¿Por qué se usa aquí?
# En tu tarea de k-means clustering, a veces aparecen advertencias como:
#     "El algoritmo no convergió completamente"
#     "Algunos clusters están vacíos"
#     "División por cero en cálculos internos"

# Es como silenciar las notificaciones de tu teléfono durante un examen. Las advertencias siguen existiendo, pero no te distraen.

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')
# Crear el modelo
model = KMeans(n_clusters=3, random_state=12345)
# Entrenar el modelo
model.fit(data)
print('Centroides de clúster:')
print(model.cluster_centers_)

# Resultado
# Centroides de clúster:
# [[40.14472236 15.00741697  8.56      ]
#  [10.90357994 29.90244865 15.096     ]
#  [10.68632155 98.90275017 10.856     ]]

# El resultado es una lista de centroides de tres vectores. 
# La primera coordenada de estos vectores contiene el tiempo promedio de sesión, 
# La segunda coordenada es el pago promedio
# La tercera coordenada es el tiempo transcurrido desde el registro del usuario.

#Observacion del datafame
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 500 entries, 0 to 499
# Data columns (total 3 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   timespent  500 non-null    float64
#  1   purchase   500 non-null    float64
#  2   months     500 non-null    float64
# dtypes: float64(3)
# memory usage: 11.8 KB

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')

# modelo con centroides de clúster
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)
print('Centroides de clúster:')
print(model.cluster_centers_)

# Modelo con centroides iniciales
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])
model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)
print('Centroides de clúster del modelo con centroides iniciales:')
print(model.cluster_centers_)

# Resultado
# Centroides de clúster:
# [[40.14472236 15.00741697  8.56      ]
#  [10.90357994 29.90244865 15.096     ]
#  [10.68632155 98.90275017 10.856     ]]

# Centroides de clúster del modelo con centroides iniciales:
# [[10.68632155 98.90275017 10.856     ]
#  [50.06201472 19.62701512  1.808     ]
#  [20.56550497 20.14513373 15.204     ]]

#---------------------------- funcion objetivo ----------------------------
# En tareas de aprendizaje supervisado, la función de pérdida muestra la diferencia entre la predicción y la respuesta correcta
# se necesita una función objetivo para evaluar la calidad del modelo.

# tenemos que encontrar los parámetros del modelo con el valor más pequeño de la función de pérdida en el conjunto de entrenamiento
# Tenemos que encontrar su valor mínimo con el conjunto de entrenamiento disponible.

# Necesitamos distribuir las observaciones de forma tal que la distancia entre ellas dentro de un clúster (distancia intraclúster) sea mínima
# La función objetivo del algoritmo se define como la suma de distancias intraclúster. 
# El propósito del entrenamiento es minimizar la suma resultante.

# el centroide de cada clúster (μ_ₖ) se calcula como el promedio para todas las observaciones en el clúster

# Con el fin de encontrar el centroide más cercano para cada observación y asignarle un número de clúster, se calcula la distancia euclidiana d₂ desde la observación hasta el centroide de cada clúster
# Luego se selecciona la diferencia más pequeña.
# El valor de ₖ en el cual se alcanza el mínimo será el número del clúster más cercano.

# Al inicio del algoritmo, los números de clúster se asignan a las observaciones de forma aleatoria
# hay una desviación intraclúster significativa (o sea que todos los puntos están dispersos y prácticamente no hay clústeres).
# Al final del algoritmo, se combinan todas las observaciones con el mismo número de clúster y los límites entre clústeres se vuelven visibles.

# La función objetivo final del algoritmo se calcula como la suma de las desviaciones intraclúster

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')
#Función objetivo
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)
print('Función objetivo:')
print(model.inertia_) #El valor de la función de pérdida está almacenado en el atributo
# La función objetivo del modelo con centroides iniciales
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])
model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)
print('La función objetivo del modelo con centroides iniciales:')
print(model.inertia_)

# Resultado
# Función objetivo:
# 68431.50999400369
# La función objetivo del modelo con centroides iniciales:
# 74253.20363562094

#---------------------------- Distancia Intraclúster ----------------------------
# La distancia intraclúster se refiere a las distancias que existen dentro de un mismo clúster. 
# Es decir, qué tan separados están los puntos que pertenecen al mismo grupo.

# ¿Qué es "Intracluster"?
# Intracluster significa "dentro del clúster". Se refiere a todo lo que ocurre al interior de cada grupo formado.

# Analogía Simple
# Imagina que tienes 3 grupos de amigos en una fiesta:
#     Distancia intraclúster: Qué tan cerca están los amigos dentro de cada grupo entre sí
#     Si los amigos de un grupo están muy juntos → distancia intraclúster pequeña (grupo compacto)
#     Si están dispersos → distancia intraclúster grande (grupo disperso)

#---------------------------- Mínimo local ----------------------------
# Tienes dos valores para la función objetivo y el valor para el modelo con centroides iniciales es más alto. 

# Tienes dos modelos con dos valores diferentes de la función objetivo:
#     Modelo sin centroides iniciales: valor más bajo (mejor)
#     Modelo con centroides iniciales: valor más alto (peor)

# ¿Por qué el modelo con centroides iniciales es peor?
# Sin centroides iniciales: n_init=10 (por defecto)
#     El algoritmo se ejecuta 10 veces con diferentes puntos de inicio aleatorios
#     Se queda con el mejor resultado de las 10 ejecuciones
# Con centroides iniciales: n_init=1 (automáticamente)
#     El algoritmo se ejecuta solo 1 vez con tus centroides específicos
#     No tiene oportunidad de "probar" otros puntos de inicio

# 10 intentos: Puedes explorar desde 10 lugares diferentes y encontrar el valle más profundo
# 1 intento: Solo exploras desde un punto, puede que no sea el mejor lugar para empezar

# El bloqueo de salida de advertencia filterwarnings() oculta el mensaje de que el número de lanzamientos del algoritmo (n_init) con centroides iniciales ahora es igual a uno.
# Función objetivo:
# 68431.50999400369
# La función objetivo del modelo con centroides iniciales:
# 74253.20363562103
# /usr/local/lib/python3.6/site-packages/sklearn/cluster/k_means_.py:969: RuntimeWarning: Posición de centro inicial explícita pasada: hacer solo un init en k-means en vez de n_init=10
#   return_n_iter=True)

# cada lanzamiento asigna al azar los números de clúster a las observaciones, de modo que obtenemos el conjunto inicial de observaciones de un clúster determinado
# En el siguiente lanzamiento, a esas mismas observaciones se les asignan nuevos números y el valor de la función objetivo cambia como resultado de la ejecución del algoritmo.

# El parámetro n_init está en 10 de forma predeterminada. Así, el algoritmo se lanza 10 veces con diferentes clústeres iniciales. 
# Se selecciona el más bajo de los mínimos locales
# De modo que la función de pérdida objetivo con centroides iniciales es mayor cuando se lanza el algoritmo solo una vez. 

# max_iter, que determina el número de iteraciones del algoritmo. 
# Cuantas más iteraciones haya, más cerca estaremos del mínimo local.

# El número predeterminado de iteraciones es 300, que es suficiente para la mayoría de las tareas.

# n_init: es responsable por el número de lanzamientos del algoritmo. Cada lanzamiento desencadena una repetición de las iteraciones.

# Si pasamos un conjunto de centroides iniciales al algoritmo k-means como entrada, el número de lanzamientos (n_init) se reducirá a 1. 
# Pocos lanzamientos hacen menos efectivo el proceso de determinar la mejor opción.

# Las reglas de relleno se configuran mediante el parámetro hue, el cual acepta una matriz de variables de string como input
# 1. transferir el número de clústeres a la matriz de strings.
# 2. nombrar los centroides para ponerlos en el gráfico

import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
data = pd.read_csv('/datasets/segments.csv')
model = KMeans(n_clusters=3, random_state=12345) #Crea un modelo KMeans para 3 clústeres
model.fit(data)
#Convierte los centroides finales en un DataFrame con las mismas columnas que tus datos originales.
centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)
data['label'] = model.labels_.astype(str) #Agrega a cada punto de datos su etiqueta de clúster (0, 1, 2)
centroids['label'] = ['0 centroid', '1 centroid', '2 centroid'] #Agrega etiquetas descriptivas a los centroides
data_all = pd.concat([data, centroids], ignore_index=True) #Combinación de Datos
# Traza el gráfico y guardarlo
pairgrid = sns.pairplot(data_all, hue='label', diag_kind='hist')

# Los segmentos básico y avanzado son más complicados. 
# Tendremos que agregar otra capa de centroides iniciales al gráfico para determinar cuáles clústeres son adecuados para ellos. 
# Pasa los valores adicionales requeridos para trazar el gráfico 

centroids_init = pd.DataFrame([[20, 80, 8], [50, 20, 5], [20, 30, 10]], #nuevos centroides                          
                            columns=data.drop(columns=['label']).columns) 
centroids_init['label'] = 4
pairgrid.data = centroids_init

# Llamemos a otro método: map_offdiag, que construye datos a partir de pairgrid.data
# El parámetro func define el tipo de gráfico
# s configura el tamaño
# marker determina la figura de los puntos 
# palette configura el color

pairgrid.map_offdiag(func=sns.scatterplot, s=200, marker='*', color='red')

# Resultado:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
data = pd.read_csv('/datasets/segments.csv')
centers = np.array([[20, 80, 8], [50, 20, 5], [20, 30, 10]])
model = KMeans(n_clusters=3, init=centers, random_state=12345)
model.fit(data)
centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)
centroids_init = pd.DataFrame(centers, columns=data.columns)
centroids_init['label'] = 4
data['label'] = model.labels_.astype(str)
centroids['label'] = ['0 centroid', '1 centroid', '2 centroid']
data_all = pd.concat([data, centroids], ignore_index=True)
pairgrid = sns.pairplot(data_all, hue='label', diag_kind='hist')
pairgrid.data = centroids_init
pairgrid.map_offdiag(func=sns.scatterplot, s=200, marker='*', palette='flag')

# el algoritmo sin centroides iniciales definió los clústeres con mínima distancia intraclúster
# mientras que los centroides iniciales los movieron.

#---------------------------- local ----------------------------
# En otras palabras, en cada lanzamiento del algoritmo, se obtiene la función objetivo como un mínimo para un conjunto inicial determinado de observaciones en el clúster.
# Al comenzar con otro conjunto inicial, el valor de la función puede cambiar. Ese será el nuevo mínimo local.
# "Local" significa que es el mejor resultado en esa zona específica, pero no necesariamente el mejor resultado en todo el espacio.

#---------------------------- ¿Qué significa "mejor" en clustering? ----------------------------
# La distancia intraclúster mínima no siempre indica el mejor resultado.

# Mínimos locales vs. globales:
# El algoritmo sin centroides iniciales puede haber encontrado un mínimo local
# Los centroides iniciales pueden haber guiado hacia una solución más representativa de los datos reales

#---------------------------- numero optimo de clusters ----------------------------
# La función objetivo del método k-means disminuye conforme el número de clústeres aumenta. 
# Si cada observación tiene su propio clúster, entonces la distancia intraclúster es igual a cero, por lo que hay que minimizar la función objetivo.
# este tipo de clustering no tiene sentido porque, básicamente, no habrá clústeres.

# los datos no siempre se distribuyen tan claramente

#---------------------------- el método de codo ----------------------------
# Para trazar un gráfico mediante este método, necesitarás hacer una lista de los valores de la función objetivo de 1 a 10 clústeres (menos de 20).

distortion = []
K = range(1, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=12345)
    model.fit(data)
    distortion.append(model.inertia_) #suma de las distancias al cuadrado de cada punto a su centroide más cercano.

# Analogía para entenderlo mejor:
#     Imagina que tienes varios grupos de personas en un parque, y cada grupo tiene un líder (centroide). 
#     La inercia sería como medir qué tan "disperso" está cada grupo
# - Imagina que tienes varios grupos de personas en un parque, y cada grupo tiene un líder (centroide). 
# - La inercia sería como medir qué tan "disperso" está cada grupo:

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.figure(figsize=(12, 8))
plt.plot(K, distortion, 'bx-')
plt.xlabel('Número de clústeres')
plt.ylabel('Valor de la función objetivo')
plt.show()

# al principio, el valor de la función objetivo disminuye abruptamente y luego se regulariza en una meseta. 
# Este momento de transición representa el número óptimo de clústeres.

#---------------------------- Ejemplo ----------------------------
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')
K = range(1, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=12345)
    model.fit(data)
    print('Número de clústeres:', k) 
    print('Valor de la función objetivo', model.inertia_)

# Resultado:
# Número de clústeres: 1
# Valor de la función objetivo 782222.4354730697
# Número de clústeres: 2
# Valor de la función objetivo 161733.64953602126
# Número de clústeres: 3
# Valor de la función objetivo 68431.50999400369
# Número de clústeres: 4
# Valor de la función objetivo 27110.79024796993
# Número de clústeres: 5
# Valor de la función objetivo 22468.766585439167
# Número de clústeres: 6
# Valor de la función objetivo 19960.565605742842
# Número de clústeres: 7
# Valor de la función objetivo 18257.632605610874

#modelo para cuatro clústeres.
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')
model = KMeans(n_clusters=4, random_state=12345)
model.fit(data)
centroids = pd.DataFrame(model.cluster_centers_, columns=data.columns)
data['label'] = model.labels_.astype(str)
centroids['label'] = ['0 centroide', '1 centroide', '2 centroide', '3 centroide']
data_all = pd.concat([data, centroids], ignore_index=True)
sns.pairplot(data_all, hue='label', diag_kind='hist')

# Muestra los centroides redondeados de los modelos resultantes
import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('/datasets/segments.csv')
# Entrenar un modelo para 3 clústeres
model = KMeans(n_clusters=3, random_state=12345)
model.fit(data)
print('Segmentos de usuarios típicos para 3 clústeres:')
print(model.cluster_centers_.round())
# Entrenar un modelo para 4 clústeres
model = KMeans(n_clusters=4, random_state=12345)
model.fit(data)
print('Segmentos de usuarios típicos para 4 clústeres:')
print(model.cluster_centers_.round())

# Resultado
# Segmentos de usuarios típicos para 3 clústeres:
# [[40. 15.  9.]
#  [11. 30. 15.]
#  [11. 99. 11.]]
# Segmentos de usuarios típicos para 4 clústeres:
# [[30. 10. 15.]
#  [11. 99. 11.]
#  [50. 20.  2.]
#  [11. 30. 15.]]
#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

#----------------------------  ----------------------------

