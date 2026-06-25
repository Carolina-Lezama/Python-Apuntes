#------------------------------Que es machine learning?-----------------------------------------
Es una herramienta para crear modelos predictivos a partir de conjuntos de datos. 
Cuantos más datos tengas, más complejos serán los programas que puedas escribir.

Los modelos aprenden de conjuntos de datos de entrenamiento. 

las filas representan observaciones y las columnas representan características.

features = características
target = objetivo

La característica que necesitamos predecir se conoce como objetivo
Tienes un conjunto de datos de entrenamiento y un objetivo que necesitas predecir usando las demas características. 

Todas las variables y características son categóricas o numéricas, 

Las tareas de clasificación se ocupan para objetivos categóricos: si solo tenemos dos categorías, es una clasificación binaria.
Si el objetivo es numérico, entonces es una tarea de regresión. 

Los datos se utilizan para encontrar relaciones entre las variables y hacer predicciones

#------------------------------clasificación binaria------------------------------------------
import pandas as pd

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1 #clasificación binaria, separada por grupos claros, en este caso una mediana, asignamos valores a una nueva clase
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

# caracteristicas: quitar aquellas columnas que no se usaran para predecir el objetivo
features = df.drop(['last_price','price_class'], axis=1)

# columna objetivo
target = df['price_class'] 


print(features.shape) #tamaño
print(target.shape)

#------------------------------Modelado-----------------------------------------
Se hacen suposiciones sobre relaciónes y luego hacen predicciones basadas en dichas suposiciones
Si las predicciones son ciertas, la suposición es correcta

Las suposiciones y los métodos de predicción se denominan modelos de machine learning.

#-----------------------------Arbol de decisiones----------------------------------------
Es un proceso de ramificacion o multicondicionales, similar a un diagrama de flujo, el valor se evalua en cada nodo con respuestas sí/no y diferentes escenarios.

Se requiere un algoritmo de aprendizaje
El conjunto de datos (de entrenamiento) se procesa a través de nuestro algoritmo de aprendizaje, produciendo un modelo entrenado.
El modelo entrenado y el algoritmo de aprendizaje dependen de un conjunto de entrenamiento.

Después del entrenamiento, el modelo está listo para hacer predicciones, sin necesidad de aprender algoritmos ni conjuntos de datos de entrenamiento.

Pasos del proceso de aprendizaje automático: entrenamiento del modelo y aplicación del modelo.

entrenamiento el modelo conoce las preguntas y respuestas. 
El modelo entrenado predice el objetivo por sí mismo.

#-----------------------------Cajas negras o algoritmos de aprendizaje----------------------------------------
from sklearn.tree import DecisionTreeClassifier   #clase para árboles de decisión

import pandas as pd
from sklearn import set_config
set_config(print_changed_only=False) # ver todos los parámetros, no solo los que cambié o aun si son por defecto

df = pd.read_csv('/datasets/train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier()  #almacena nuestro modelo vacio, y tenemos que ejecutar un algoritmo de aprendizaje para entrenarlo
model.fit(features, target) #iniciar el entrenamiento, aprende la relación entre las variables de entrada y las variables objetivo

print(model)
#Salida:
# DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#    max_depth=None, max_features=None, max_leaf_nodes=None,
#    min_impurity_decrease=0.0, min_samples_leaf=1,
#    min_samples_split=2, min_weight_fraction_leaf=0.0,
#    random_state=None, splitter='best')

new_features = pd.DataFrame(
    [
        [None, None, 2.8, 25, None, 25, 0, 0, 0, None, 0, 30706.0, 7877.0],
        [None, None, 2.75, 25, None, 25, 0, 0, 0, None, 0, 36421.0, 9176.0]
    ],
    columns=features.columns
)
new_features.loc[0, 'total_area'] = 900.0
new_features.loc[1, 'total_area'] = 109
new_features.loc[0, 'bedrooms'] = 12
new_features.loc[1, 'bedrooms'] = 2
new_features.loc[0, 'living_area'] = 409.7
new_features.loc[1, 'living_area'] = 32
new_features.loc[0, 'kitchen_area'] = 112
new_features.loc[1, 'kitchen_area'] = 40.5

# predice respuestas y muestra el resultado en la pantalla
answers = model.predict(new_features)
print(answers)

El conjunto de datos de entrenamiento se almacena en las variables features y target.
Las características de las nuevas observaciones se registran en la variable new_features, es elconjunto de datos que queremos predecir.
El método fit() se usa para entrenamiento, y el método predict() se usa para prueba.

#-----------------------------Aleatoriedad en algoritmos de aprendizaje----------------------------------------
Agregar aleatoriedad a los algoritmos de aprendizaje es util para la capacidad de un modelo para descubrir relaciones en los datos. 
El barajado introduce aleatoriedad en el algoritmo de aprendizaje.
Una computadora no puede generar números verdaderamente aleatorios. Utiliza generadores de números pseudoaleatorios que crean secuencias aparentemente aleatorias.
















el orden de las tarjetas parecerá aleatorio pero sigue siendo el mismo solo que ordenado de otra manera
Para que el algoritmo de aprendizaje utilice siempre los mismos números pseudoaleatorios, simplemente especifica el parámetro random_state:

# especifica un estado aleatorio (número)
model = DecisionTreeClassifier(random_state=12345)
# Puedes establecer cualquier valor para random_state, lo importante es utilizar el mismo valor durante todo el tiempo que desees obtener exactamente el mismo modelo.
# Si configuras random_state=None (el valor predeterminado), la pseudoaleatoriedad siempre será diferente.

# entrena al modelo de la misma manera que antes
model.fit(features, target)

#-----------------------------Predicciones correctas e incorrectas---------------------------------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data_us.csv')

test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)

def error_count(answers, predictions):
    contador = 0
    for i in range(len(answers)): 
        if answers[i] != predictions[i]:
            contador +=1
    return contador

print('Errores:', error_count(test_target, test_predictions))

#Predicciones: [1. 0. 0.]
# Respuestas correctas: [0. 1. 0.]

#Errores: 275

#-----------------------------Exactitud en la prediccion---------------------------------------
#el número de errores nos importa solo en relación con el número total de respuestas
exactitud = (len(test_target) - error_count(test_target, test_predictions)) / len(test_target)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345, class_weight='balanced')

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data_us.csv')

test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)


def error_count(answers, predictions):
    count = 0
    for i in range(len(answers)):
        if answers[i] != predictions[i]:
            count += 1
    return count

def accuracy(answers, predictions):
    correct = 0
    for i in range(len(answers)):
        if answers[i] == predictions[i]:
                correct +=1
    return correct / len(answers)
print('Accuracy:', accuracy(test_target, test_predictions))

#Accuracy: 0.6481481481481481

#-----------------------------Métricas de evaluación---------------------------------------
se utilizan para medir la calidad de un modelo
exactitud: número de predicciones correctas dividido por el número total de predicciones
precisión: número de verdaderos positivos dividido por el número total de predicciones positivas
recall (sensibilidad): número de verdaderos positivos dividido por el número total de casos positivos reales

precisión (precision) toma todos los apartamentos que el modelo consideró caros (están marcados como "1") y calcula qué fracción de ellos era costosa realmente. Los apartamentos no reconocidos por el modelo se ignoran.
recall (sensibilidad) toma todos los apartamentos que son realmente caros y calcula qué fracción de ellos reconoció el modelo. Los apartamentos que fueron reconocidos por el modelo por error se ignoran.

Digamos que tienes un trabajo en un banco. El banco tiene 10 000 clientes. Sin embargo, 100 de ellos son estafadores, y tu trabajo es rastrearlos. El Clasificador A marca a 1000 clientes como posibles estafadores, pero solo 90 de ellos son estafadores reales. El Clasificador A tiene un recall alto (90%) porque logra encontrar a casi todos los estafadores, pero su precisión es baja (9%) ya que acusa erróneamente a cientos de ciudadanos honrados. Por otro lado, el Clasificador B marca a 10 clientes como estafadores, y 9 de ellos son realmente estafadores. Su recall es menor porque solo puede encontrar el 9% de ellos, pero la precisión es del 90% (¡9 de 10!).

En otras palabras, recall es una prioridad cuando es importante encontrar todas las observaciones requeridas, incluso a costa de cometer muchos errores, y la precisión es importante cuando tu objetivo es minimizar los errores, incluso si eso significa que muchos casos no se detectan.

Diferentes tareas requieren diferentes métricas de evaluación. ¿Por qué elegimos exactitud al determinar los precios de los apartamentos? Porque necesitamos tener en cuenta ambos tipos de errores (ceros que deberían ser unos, así como unos que deberían ser ceros), y eso es justo lo que hace la exactitud. Cada predicción incorrecta conduce a una recomendación incorrecta y a una posible pérdida de ganancias para el vendedor. A su vez, una mayor exactitud en la clasificación genera más ganancias. 

Siempre asegúrate de que tu modelo funcione mejor que la casualidad, es decir, realiza una prueba de cordura.













#-----------------------------Métricas de evaluación en Scikit-Learn---------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score    #importancion de funciones
accuracy = accuracy_score(target, predictions)  #recibe respuestas correctas ylas predicciones del modelo: devuelve exactitud(valor)

#Ejemplo completo:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/datasets/train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0
features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']
model = DecisionTreeClassifier(random_state=12345)
model.fit(features, target) #entrnamiento

test_df = pd.read_csv('/datasets/test_data_us.csv')
test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0
test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']

train_predictions = model.predict(features) #predicciones del conjunto de entrenamiento
test_predictions = model.predict(test_features) #predicciones del conjunto de prueba

print('Exactitud')
print('Training set:', accuracy_score(target, train_predictions))  #exactitud del conjunto de entrenamiento
print('Test set:', accuracy_score(test_target, test_predictions))  #exactitud del conjunto de prueba

#-----------------------------Sobreajuste y subajuste ---------------------------------------
Sobreajuste:
Cuando un modelo aprende demasiado bien los datos de entrenamiento, incluidos los errores y el ruido, y no puede generalizar bien a datos nuevos.
intenta categorizar perfectamente un conjunto de datos particular
Entrena demasiado un modelo y detectará las fluctuaciones y desviaciones que provienen del ruido aleatorio, en lugar de las relaciones de causa y efecto realmente existentes
Un modelo sobreajustado comenzará a ver dependencias que en realidad no existen. 
Intentará aplicar estas dependencias inexistentes a nuevos datos que podrían clasificarse mediante una regla más simple y general.

Subajuste :
Ocurre cuando la exactitud es baja y aproximadamente igual tanto para el conjunto de entrenamiento como para el de prueba.

Una profundidad de árbol alta significa que el modelo tiende a sobreajustarse. Un árbol corto puede conducir a un subajuste.

En arboles de decisiones:
Una profundidad de árbol alta significa que el modelo tiende a sobreajustarse. Un árbol corto puede conducir a un subajuste.

#-----------------------------Establecer produnfidad de un arbol ---------------------------------------
# especificar la profundidad (ilimitado por defecto)
model = DecisionTreeClassifier(random_state=12345, max_depth=3)

model.fit(features, target)

#Ejemplo completo:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

test_df = pd.read_csv('/datasets/test_data_us.csv')

test_df.loc[test_df['last_price'] > 113000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 113000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']

for depth in range(1, 11):  # ciclo para la profundidad
    model = DecisionTreeClassifier(random_state=54321, max_depth=depth)
    model.fit(features, target)  #entrenamiento
    
    train_predictions = model.predict(features)  #predicciones
    test_predictions = model.predict(test_features)

    print("Exactitud de max_depth igual a", depth)
    print("Conjunto de entrenamiento:", accuracy_score(target, train_predictions)) # calcula la puntuación de accuracy en el conjunto de entrenamiento
    print("Conjunto de prueba:", accuracy_score(test_target, test_predictions)) # calcula la puntuación de accuracy en el conjunto de prueba
    print()

#resultados:
Exactitud de max_depth igual a 1
Conjunto de entrenamiento: 0.8435719784449577
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 2
Conjunto de entrenamiento: 0.8435719784449577
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 3
Conjunto de entrenamiento: 0.8665127020785219
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 4
Conjunto de entrenamiento: 0.8739030023094688
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 5
Conjunto de entrenamiento: 0.891301000769823
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 6
Conjunto de entrenamiento: 0.9023864511162433
Conjunto de prueba: 0.5092592592592593

Exactitud de max_depth igual a 7
Conjunto de entrenamiento: 0.9140877598152425
Conjunto de prueba: 0.37808641975308643

Exactitud de max_depth igual a 8
Conjunto de entrenamiento: 0.930715935334873
Conjunto de prueba: 0.3950617283950617

Exactitud de max_depth igual a 9
Conjunto de entrenamiento: 0.9441108545034642
Conjunto de prueba: 0.4367283950617284

Exactitud de max_depth igual a 10
Conjunto de entrenamiento: 0.957197844495766
Conjunto de prueba: 0.3950617283950617

A medida que aumenta la profundidad máxima del árbol, también lo hace la exactitud en el conjunto de entrenamiento. Sin embargo, la exactitud deja de mejorar cuando se alzanza 5, 
Estos empeoran a medida que se es mas profundo

#-----------------------------Dataset de validacion---------------------------------------
Para que el control de calidad sea fiable necesitamos un dataset de validación.
El dataset de validación se separa del dataset fuente antes de que se entrene el modelo. 
La validación muestra cómo se comporta el modelo en el campo y ayuda a revelar si hay sobreajuste.

La parte de los datos que se va a asignar al conjunto de validación depende del número de observaciones y características, así como de la variación de los datos.

1) El conjunto de prueba existe (o existirá en el futuro cercano) pero no está disponible por el momento:
    La proporción ideal es 3:1. Esto significa que un 75 % es para el conjunto de entrenamiento y un 25 % es para el conjunto de validación.

2) El conjunto de prueba no existe:
    la proporción es 2:1:1. Esto significa que un 50 % es para el conjunto de entrenamiento, un 25 % para el conjunto de validación y un 25 % para el conjunto de prueba.
    la proporción es 3:1:1. Esto significa que un 60 % es para el conjunto de entrenamiento, un 20 % para el conjunto de validación y un 20 % para el conjunto de prueba.

La validación muestra cómo lidia el modelo con datos desconocidos. Si la exactitud de la predicción cae dramáticamente, entonces el modelo está sobreajustado.
El dataset de validación se usa para comprobar la calidad del modelo y configurarlo.
Se usa el conjunto de prueba para la evaluación final del modelo entrenado.

#-----------------------------Separacion de datos--------------------------------------
from sklearn.model_selection import train_test_split  #puede separar cualquier conjunto de datos en dos: entrenamiento y prueba. Pero nosotros lo usaremos para obtener un conjunto de validación y uno de entrenamiento.

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)

Ejemplo:
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345) #entranamiento y validacion

# < declara variables para las características y para la característica objetivo >
features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

print(features_train.shape)
print(target_train.shape)
print(features_valid.shape)
print(target_valid.shape)

Salida:
(4871, 13) #75%
(4871,)
(1624, 13) #25%
(1624,)

#-----------------------------Hiperparámetros--------------------------------------
Estos son configuraciones para algoritmos de aprendizaje. Se pueden ajustar antes del entrenamiento.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data_us.csv')
df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345)

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

for depth in range(1,6):
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345) #max_depth es un hiperparámetro
    model.fit(features_train, target_train)  
    predictions_valid = model.predict(features_valid)  
    print("max_depth =", depth, ": ", end='')
    print(accuracy_score(target_valid, predictions_valid))

Resultados:
max_depth = 1 : 0.8522167487684729
max_depth = 2 : 0.8522167487684729
max_depth = 3 : 0.8466748768472906
max_depth = 4 : 0.8725369458128078 #verificar cual depth es mejor para evitar sobreajuste
max_depth = 5 : 0.8663793103448276

#-----------------------------bosque aleatorio--------------------------------------
Este algoritmo entrena una gran cantidad de árboles independientes y toma una decisión mediante el voto. 
El voto es para obtener una valoración promedio que anule el sesgo personal y los errores
Un bosque aleatorio ayuda a mejorar los resultados y a evitar el sobreajuste.

from sklearn.ensemble import RandomForestClassifier
hiperparámetro n_estimators: establecer el número de árboles en el bosque.
El aumento en la cantidad de estimadores siempre disminuye la varianza de la predicción, mientras más árboles uses mejores resultados obtendrás.

model = RandomForestClassifier(random_state=54321, n_estimators=3) #100 arboles por defecto
model.fit(features, target)
model.score(features, target) #en lugar de predict y accuracy_score

Ejemplo completo:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=54321) # segmenta el 25% de los datos para hacer el conjunto de validación

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

best_score = 0
best_est = 0
for est in range(1, 31): # selecciona el rango del hiperparámetro
    model = RandomForestClassifier(random_state=54321, n_estimators=est) # configura el número de árboles
    model.fit(features_train,target_train) # entrena el modelo en el conjunto de entrenamiento
    score = model.score(features_valid,target_valid) # calcula la puntuación de accuracy en el conjunto de validación
    if score > best_score:
        best_score = score # guarda la mejor puntuación de accuracy en el conjunto de validación
        best_est = est # guarda el número de estimadores que corresponden a la mejor puntuación de exactitud

print("La exactitud del mejor modelo en el conjunto de validación (n_estimators = {}): {}".format(best_est, best_score))

Resultados:
La exactitud del mejor modelo en el conjunto de validación (n_estimators = 29): 0.8885467980295566

#-----------------------------regresión logística--------------------------------------
Por lo general, el eje Y tiene un rango de 0 a 1 y representa la probabilidad de un evento. El eje X traza el gráfico de una variable que afecta a la probabilidad.
la regresión logística puede hacer una categorizacion binaria
Si la curva y el área determinada nos dan una probabilidad de más de 0.5 llevara 1 sino 0
La regresión logística puede usarse con más de una característica a la vez

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=54321, solver='liblinear')

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=54321) # segmenta el 25% de los datos para hacer el conjunto de validación

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

model = LogisticRegression(random_state=54321, solver='liblinear')
# inicializa el constructor de regresión logística con los parámetros random_state=54321 y solver='liblinear'

model.fit(features_train, target_train) # entrena el modelo en el conjunto de entrenamiento
score_train = model.score(features_train, target_train) # calcula la puntuación de accuracy en el conjunto de entrenamiento
score_valid = model.score(features_valid,target_valid) # calcula la puntuación de accuracy en el conjunto de validación

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)

#----------------Comparar arbol de decision, regresion logistica y bosque aleatorio--------------------------------------
Exactitud:
    1.bosque aleatorio tiene la exactitud más alta porque usa un conjunto de árboles en vez de uno.
    2.regresión logística. El modelo no es complicado así que no habrá sobreajuste.
    3.Los árboles de decisión tienen la menos exactitud. Si la profundidad es menos de 4, está subajustado; si es mayor a 4, está sobreajustado.

Velocidad de ejecución:
    1.regresión logística es la más rápida, tiene menor número de parámetros.
    2.árboles de decisión también es alta y depende de la profundidad del árbol
    3.bosque aleatorio es el más lento: cuantos más árboles haya, más lento trabaja este modelo.

#---------------------------------------Regresion-------------------------------------
Un objetivo (respuesta) puede ser categórico o numérico. 
Un objetivo categórico indica un problema de tipo clasificación. 
Si el objetivo es numérico, la tarea es regresión. 

#-------------------------------------Error cuadrático medio------------------------------------
el EMC debe ser lo más bajo posible.    
Si tomas la raíz cuadrada del EMC obtendrás una métrica llamada RECM (en inglés RMSE), que significa raíz del error cuadrático medio. Tiene las mismas unidades de medida que la variable objetivo.

# Ejemplo manual
import numpy as np
valores_reales = [100, 150, 200, 120, 180]
predicciones = [95, 160, 190, 125, 175]
def calcular_emc(reales, predicciones):
    # Paso 1: Calcular diferencias
    diferencias = []
    for i in range(len(reales)):
        diferencia = reales[i] - predicciones[i]
        diferencias.append(diferencia)
        print(f"Real: {reales[i]}, Predicción: {predicciones[i]}, Diferencia: {diferencia}")
    # Paso 2: Elevar al cuadrado
    cuadrados = []
    for diferencia in diferencias:
        cuadrado = diferencia ** 2
        cuadrados.append(cuadrado)
        print(f"Diferencia: {diferencia}, Cuadrado: {cuadrado}")
    # Paso 3: Calcular promedio
    suma_cuadrados = sum(cuadrados)
    emc = suma_cuadrados / len(cuadrados)
    print(f"Suma de cuadrados: {suma_cuadrados}")
    print(f"Número de observaciones: {len(cuadrados)}")
    print(f"EMC = {suma_cuadrados} / {len(cuadrados)} = {emc}")
    return emc
emc_manual = calcular_emc(valores_reales, predicciones)
print(f"EMC final: {emc_manual}")

# Ejemplo con scikit-learn
from sklearn.metrics import mean_squared_error
valores_reales = [100, 150, 200, 120, 180]
predicciones = [95, 160, 190, 125, 175]
emc_sklearn = mean_squared_error(valores_reales, predicciones)
print(f"EMC con sklearn: {emc_sklearn}")

#---------------------------------Ejercicio------------------------------------
import pandas as pd
from sklearn.metrics import mean_squared_error
df = pd.read_csv('/datasets/train_data_us.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price']
predictions = pd.Series(target.mean(), index=target.index)

mse = mean_squared_error(target, predictions) #solo acepta matrices; ambos con el mismo tamaño aunque sea igual para todos
rmse = mse ** 0.5 #sacar raiz cuadrada

mse o Error Cuadrático Medio:
    mide qué tan "lejos" están nuestras predicciones de los valores reales.
    Si una casa vale $200,000 y predices $180,000, el error es $20,000
    El MSE toma este error, lo eleva al cuadrado ($20,000)² y luego promedia todos estos errores cuadrados

rmse o Raíz del Error Cuadrático Medio:
    MSE te da "dólares cuadrados" - una unidad que no tiene sentido práctico.
    Al sacar la raíz cuadrada, vuelves a las unidades originales (dólares normales).

#----------------------Arboles de decision y regresion---------------------------
predicen un numero
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data_us.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

best_model = None
best_result = 10000
best_depth = 0
for depth in range(1, 6): 
    model = DecisionTreeRegressor(random_state=12345, max_depth=depth)
    model.fit(features_train, target_train) 
    predictions_valid = model.predict(features_valid) # obtén las predicciones del modelo en el conjunto de validación
    result = mean_squared_error(target_valid, predictions_valid)**0.5
    if result < best_result:
        best_model = model
        best_result = result
        best_depth = depth

print(f"RECM del mejor modelo en el conjunto de validación (max_depth = {best_depth}): {best_result}")

#----------------------Bosque aletorio y regresion---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data_us.csv')

df_train, df_valid = train_test_split(df, random_state=54321, test_size = 0.25)

features_train = df_train.drop(['last_price'], axis=1)
target_train =  
features_valid = df_valid.drop(['last_price'], axis=1)
target_valid = df_valid['last_price'] / 100000

best_error = 10000 # configura el inicio de RECM
best_est = 0
best_depth = 0
for est in range(10, 51, 10):
    for depth in range (1, 11):
        model = RandomForestRegressor(random_state=54321, n_estimators=est, max_depth=depth)
        model.fit(features_train, target_train)
        predictions_valid = model.predict(features_valid) # obtén las predicciones del modelo en el conjunto de validación
        error = mean_squared_error(target_valid, predictions_valid)**0.5
        print("Validación RECM para los n_estimators de", est, ", depth=", depth, "is", error)
        if error < best_error: # guardamos la configuración del modelo si se logra el error más bajo
            best_error = error
            best_est = est
            best_depth = depth

print("RECM del mejor modelo en el conjunto de validación:", best_error, "n_estimators:", best_est, "best_depth:", best_depth)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data_us.csv')

# inicializa las variables
features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 100000

final_model = RandomForestRegressor(random_state=54321, n_estimators=50, max_depth=10)
final_model.fit(features,target)

predictions = final_model.predict(features)
rmse = mean_squared_error(target, predictions)**0.5
print(rmse)

#-------------------------regresion logistica ---------------------------
hay una sencilla ecuación lineal que describe cómo un determinado parámetro define el output
El algoritmo de regresión lineal determina los valores óptimos para a y b de manera que traza la línea lo más cerca posible de los puntos (para eso, minimiza el ECM)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data_us.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345) 

model = LinearRegression()
model.fit(features_train,target_train) # entrena el modelo en el conjunto de entrenamiento
predictions_valid = model.predict(features_valid) # obtén las predicciones del modelo en el conjunto de validación

result = mean_squared_error(target_valid, predictions_valid)**0.5
print("RECM del modelo de regresión lineal en el conjunto de validación:", result)

