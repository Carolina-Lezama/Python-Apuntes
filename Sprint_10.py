#-----------------------------Repaso--------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  #para tareas de clasificacion
data = pd.read_csv('/datasets/travel_insurance_us.csv')
train, valid = train_test_split(data, test_size=0.25, random_state=12345)
features_train = train.drop('Claim', axis=1)
target_train = train['Claim']
features_valid = valid.drop('Claim', axis=1)
target_valid = valid['Claim']
print(features_train.shape)
print(features_valid.shape)

#La regresión logística determina la categoría utilizando una fórmula que consta de características numéricas. si hay strings causara errores

#------------------------transformar características categóricas en numéricas--------------------------------------
print(data['Gender'].head() ) #mostrar en formato original
    # 0       M
    # 1    None
    # 2       F
    # 3    None
    # 4       M

print(pd.get_dummies(data['Gender']).head()) #convierte la variable categórica en variables numéricas dummy.
# Las nuevas columnas se denominan variables dummy(orma de convertir texto en números)
    #    F  M  None
    # 0  0  1     0
    # 1  0  0     1
    # 2  1  0     0
    # 3  0  0     1
    # 4  0  1     0

#-----------------------La trampa dummy, confundir al modelo-------------------------------------
print(pd.get_dummies(data['Gender'] , drop_first=True ).head()) #Se eliminó la primera columna, evitar redundancia
#si una es 0 y otra es 0, la tercera tiene que ser 1
# Resultado, eliminamos columas para evitar confundir al modelo
#    M  None
# 0  1     0
# 1  0     1
# 2  0     0
# 3  0     1
# 4  1     0

data_ohe = pd.get_dummies(data,drop_first=True ) #Convierte TODAS las columnas categóricas del dataframe en variables numéricas dummy
#si son numericas se mantienen.
#OHE = One Hot Encoding (Codificación One-Hot)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('/datasets/travel_insurance_us.csv')
data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
print('¡Entrenado!')

#-----------------------Codificacion de etiquetas-------------------------------------
# One Hot Encoding (OHE)
# Crea múltiples columnas (una por cada categoría)
# Cada columna contiene valores 0 o 1
# Mantiene la independencia entre categorías
# No asume orden entre las categorías
# Ideal para regresión logística y algoritmos lineales

# Codificación de etiquetas
# Usa una sola columna con números
# Asigna números arbitrarios a cada categoría (0, 1, 2, etc.)
# Puede crear relaciones falsas entre categorías
# El algoritmo puede interpretar que existe un orden donde no lo hay
# Mejor para árboles de decisión y algoritmos basados en árboles

# se utiliza para codificar características categóricas para árboles de decisión y otros algoritmos basados en árboles, como bosques aleatorios
f > 2
m > 1
none > 0

from sklearn.preprocessing import OrdinalEncoder #para una o más columnas a la vez
encoder = OrdinalEncoder() #mismas variables en entreno y transformacion
encoder.fit(data) #que reconozca qué variables son categóricas
data_ordinal = encoder.transform(data) #transformar y guardar los datos; Devuelve un array de NumPy

data_ordinal = pd.DataFrame(encoder.transform(data), columns=data.columns) #nuevo df con variables numericas
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns) #combina fit() y transform(); nuevo df

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
data = pd.read_csv('/datasets/travel_insurance_us.csv')
encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns )
print(data_ordinal.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
data = pd.read_csv('/datasets/travel_insurance_us.csv')
encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)
target = data_ordinal['Claim']
features = data_ordinal.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train) #entrenado con codificación de etiquetas
print('¡Entrenado!')

#-----------------Diferencias entre codificación ordinal de codificación de etiquetas-------------------------------------
# odificación de etiquetas: no hay forma de decidir como se etiquetaban las categorías(forma arbitraria).
# Caso hipotetico:
# tienes una variable que contenga descripciones de temperatura: "caliente", "tibia" y "fría". 
# Estas tienen un orden logico, por lo que asignarles etiquetas de forma arbitraria sería un error. 
# Si codificas "fría" con 0, "caliente" con 1 y "tibia" con 2, entonces el algoritmo funcionará suponiendo que "caliente" es una cualidad entre "fría" y "tibia", en lugar de una cualidad aún más cálida que "tibia".

# codificación ordinal: etiquetas numéricas dispuestas en un orden específico, generalmente enumeración manual.
# codificación nominal: una variable de categorías sin orden

# Codificacion ordinal con pandas:
temperature_dict = {'cold': 0, 'warm': 1, 'hot': 2}
df['temperature'] = df['temperature'].map(temperature_dict)

quality_dict = {'malo': 0, 'bueno': 1, 'excelente': 2}
df['calidad'] = df['calidad'].map(quality_dict)

# Codificacion ordinal con sklearn:
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['malo', 'bueno', 'excelente']]) #especificar el orden exacto en categories
df['calidad_encoded'] = encoder.fit_transform(df[['calidad']])

# Codificacion ordinal con category_encoders:
from category_encoders import OrdinalEncoder
encoder = OrdinalEncoder(mapping=[{
    'col': 'calidad',
    'mapping': {'malo': 0, 'bueno': 1, 'excelente': 2}
}])
df_encoded = encoder.fit_transform(df)

#----------------elegir método correcto y porque codificación de etiquetas no es util en regresión logística-------------------------------------
# la codificación de etiquetas es una mala idea, especialmente para los algoritmos de regresión
# Si "gato" es 0 y "perro" es 1, no queremos que nuestro modelo piense que "perro" es de alguna manera mayor que "gato", 
# Tampoco queremos que el modelo use el mismo peso para varias entidades completamente diferentes.

# en caso de regresión, las categorías deben codificarse de una manera que les asigne la misma importancia a todas, pero que aún así se reconozcan como distintas y diferentes

# arboles: codificacion de etiquetas
# de regresion: OHE (One-Hot Encoding)    

# cuantas más categorías tenga que codificar la variable, mayores serán los inconvenientes de OHE
# Las variables que tienen muchas categorías se denominan variables de alta cardinalidad, y el uso de OHE para codificarlas conduce a un alto consumo de memoria, escasez de datos y otros problemas típicamente asociados con la alta dimensionalidad de los datos

#-----------------Escalado de caracteristicas-------------------------------------
# Estandarizar los datos, con el escalado de caracteristicas
# si entre 2 columnas, la magnitud de los valores y la dispersión es mayor, el algoritmo encontrará que una característica es más importante que otra
# Todas las características deben considerarse igual de importantes antes de la ejecución del algoritmo.

#Ejemplo:
Age = [25, 30, 45, 60] # Edad: valores entre 18-65
Commission = [5000, 15000, 25000, 45000] # Comisión: valores entre 1000-50000  

# Mayor magnitud: números más grandes (miles vs decenas)
# Mayor dispersión: rango más amplio (49,000 vs 47)

# "Commission debe ser MÁS IMPORTANTE"
# Pero esto es FALSO - solo es una diferencia de escala, no de importancia.
# Con StandardScaler(Estandarización) convertimos todo a la misma escala

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #funcion
data = pd.read_csv('/datasets/travel_insurance_us.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']
scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
print(features_train.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('travel_insurance_us.csv')
data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
numeric = ["Duration", "Net Sales", "Commission (en valor)", "Age"]
scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
print(features_train.shape) #muestra las dimensiones de tu conjunto de datos

#preprocesamiento completo de datos
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('/datasets/flights.csv')
data_ohe = pd.get_dummies(data, drop_first=True) #convertir variables categóricas en numéricas
target = data_ohe['Arrival Delay']
features = data_ohe.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
numeric = ['Day', 'Day Of Week', 'Origin Airport Delay Rate',
       'Destination Airport Delay Rate', 'Scheduled Time', 'Distance',
       'Scheduled Departure Hour', 'Scheduled Departure Minute']
scaler = StandardScaler() #Estandarización de características numéricas
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
print(features_train.shape)
print(features_valid.shape)

#---------------------------------Prueba de consistencia-------------------------------------
# verificar si las clases de la característica objetivo están equilibradas o desequilibradas
# asegurarnos de que nuestro preprocesamiento funcionó correctamente.

import pandas as pd
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
class_frequency = data['Claim'].value_counts(normalize=True) #Contar cada valor unico en la columna objetivo
# Con normalize=True (proporciones):
# No     0.714  (5/7 = 71.4%)
# Yes    0.286  (2/7 = 28.6%)
# Sin normalize (conteos absolutos):
# No     5
# Yes    2
print(class_frequency)
class_frequency.plot(kind='bar')

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = pd.Series(model.predict(features_valid)) # para hacer que value_counts() funcione, convertimos los resultados a pd.Series
class_frequency = predicted_valid.value_counts(normalize=True) 
print(class_frequency)
class_frequency.plot(kind='bar')

#modelo constante: siempre predice la misma clase (la mayoritaria)
import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
target_pred_constant = pd.Series(0, index=target.index) #crear una serie con solo valores 0, Usa exactamente los mismos índices que tu variable objetivo original
print(accuracy_score(target, target_pred_constant))

#---------------------------------Equilibrio y desequilibrio de las clases------------------------------------
# Modelo constante: modelo basico que no aprende nada, siempre predice la misma respuesta
# Un modelo constante diría: "Para TODOS los usuarios, recomiendo plan Smart"
# Resultado: 69.4% de exactitud (porque acertaría solo con los que realmente tienen Smart)

# Si tu modelo de machine learning no supera al modelo constante, ¡algo está muy mal!

# desequilibrio de clases: tu modelo puede "hacer trampa" y simplemente predecir siempre la clase mayoritaria
# Tendra mucha exactitud, pero no aprendera nada util, cuando se enfrente a datos nuevos, fallara

# Las clases están desequilibradas cuando su proporción está lejos de 1:1. (ej: 9,500 vs 500)
# El equilibrio de clases se observa si su número es aproximadamente igual. (ej: 5,000 vs 5,000)

# Una clase con la etiqueta "1" se llama positiva 
# una clase con la etiqueta "0" se llama negativa.

# Respuestas de verdadero positivo (VP 1/1) y verdadero negativo (VN 0/0) 
# Respuestas de falso positivo (FP modelo=1 realidad=0) y falso negativo (FN modelo=0 realidad=1)
# "positivo" y "negativo" se refieren a una predicción
# "verdadero" y "falso" se refieren a su exactitud.

import pandas as pd
target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# En el ejemplo la respuesta Verdadero Positivo equivale a:
#     pidió compensación, de acuerdo con la predicción del modelo (1)
#     sí hizo un reclamo de seguro. (1)
print(((target == 1) & (predictions == 1)).sum())

# En el ejemplo la respuesta Verdadero Positivo equivale a:
#     de acuerdo con la predicción del modelo, no solicitaron un pago
#     en realidad no solicitaron la compensación del seguro.
print(((target == 0) & (predictions == 0)).sum())

# En el ejemplo la respuesta Falso Positivo equivale a:
#     según la predicción del modelo, solicitaron un pago, pero
#     en realidad, no hicieron un reclamo.
print(((target == 0) & (predictions == 1)).sum())

# En el ejemplo la respuesta Falso Positivo equivale a:
#     según la predicción del modelo, no solicitaron un pago, pero
#     en realidad, hicieron un reclamo.
print(((target == 1) & (predictions == 0)).sum())

#---------------------------------matriz de confusión------------------------------------
# Cuando VP, FP, VN, FN se recopilan en una tabla:
#     eje horizontal ("Predicciones") (0 y 1)
#     eje vertical ("Respuestas" , "respuestas reales") (0 y 1)

# Predicciones correctas:
#     VN en la esquina superior izquierda
#     VP en la esquina inferior derecha

# Predicciones incorrectas:
#     FP en la esquina superior derecha
#     FN en la esquina inferior izquierda

# La matriz de confusión te permite visualizar los resultados de calcular las métricas de exactitud y recall.

import pandas as pd
from sklearn.metrics import confusion_matrix #funcion
target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
print(confusion_matrix(target,predictions))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix #funcion
from sklearn.model_selection import train_test_split
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid) #tiene las predicciones, las compararemos con las reales
print(confusion_matrix(target_valid,predicted_valid))

# porque usar datos de valid y no de train (target_valid y predicted_valid):
#     queremos evaluar el modelo con datos que NUNCA ha visto antes.
#     compararemos las predicciones con las respuestas reales para esos datos nuevos.

#---------------------------------Recall------------------------------------
# Recall = ¿De todos los casos positivos reales, cuántos logré encontrar? revela la porción de respuestas positivas identificadas por el modelo
# Recall = VP / (VP + FN)
# Donde:
# - VP = Verdaderos Positivos (aciertos positivos)
# - FN = Falsos Negativos (me equivoqué, era positivo pero dije negativo)
# Tienes 100 emails spam reales en tu bandeja:
# Escenario 1:
# - Tu modelo detecta 80 como spam ✅ (VP = 80)
# - Se le escapan 20 spam ❌ (FN = 20)
# - Recall = 80/(80+20) = 80%

# cerca del 1: el modelo es bueno para identificar verdaderos positivos
# cerca del 0: el modelo necesita ser revisado y arreglado.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score #funcion
from sklearn.model_selection import train_test_split
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print(recall_score(target_valid,predicted_valid))

#---------------------------------precisión------------------------------------
# La precisión mide la relación entre las predicciones verdaderas positivas y todas las predicciones positivas realizadas por el modelo. Por lo tanto, cuantas más predicciones de falsos positivos se hagan, menor será la precisión.
# Precisión = Verdaderos Positivos / (Verdaderos Positivos + Falsos Positivos)
# Verdaderos Positivos (VP): El modelo predijo "va a reclamar" y la persona SÍ reclamó → 80 casos
# Falsos Positivos (FP): El modelo predijo "va a reclamar" pero la persona NO reclamó → 20 casos
# - Si tienes 20 falsos positivos → Precisión = 80/100 = 80%
# - Si tienes 40 falsos positivos → Precisión = 80/120 = 67%

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score #funcion
from sklearn.model_selection import train_test_split
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print(precision_score(target_valid,predicted_valid))

#--------------------------------Recall vs. Precisión------------------------------------
# ¿cómo puedes hacer que sea lo más alta posible? Entrena un modelo que prediga "1" para todas las observaciones. Recall será igual a 1.0
# Si el modelo predice "1" para todas las observaciones, el valor de FN será 0, lo que dejará solo VP en el denominador de la fórmula, por lo que recall será 1 (VP/VP).
# La fórmula tiene en cuenta solo el error de la clase positiva, pero ignora la clase negativa. Debes entrenar un modelo que responda "1" con la menor frecuencia posible.

# Imagina que tienes 100 personas y 30 de ellas realmente van a reclamar un seguro (casos positivos reales).
# ### Escenario normal:
# - El modelo predice correctamente 20 casos → VP = 20
# - El modelo se equivoca y dice "no reclamarán" para 10 casos que sí reclamarán → FN = 10
# - Recall = 20 / (20 + 10) = 20/30 = 0.67
# ### Escenario "trampa" - modelo que predice "1" para TODO:
# - El modelo dice "SÍ reclamará" para las 100 personas
# - De las 30 que realmente reclamarán, las identifica todas → VP = 30
# - No se pierde ningún caso positivo → FN = 0
# - Recall = 30 / (30 + 0) = 30/30 = 1.0
# Pero cuidado: Este modelo sería terrible en precisión porque tendría muchísimos falsos positivos; no sirve en la vida real

#--------------------------------Valor F1 (relación de recall a precisión es 1:1)------------------------------------
# Recall y precision evalúan la calidad de las predicciones de la clase positiva desde diferentes ángulos. 

# Recall describe qué tan bien comprende un modelo las propiedades de esta clase y es capaz de reconocerla. 
# Recall: "¿Qué tan bueno soy detectando TODOS los incendios?"
# Pregunta clave: De todos los incendios reales, ¿cuántos logro detectar?
# Miedo principal: Que se me escape un incendio (Falso Negativo)
# Ejemplo: Si hay 10 incendios reales y detecto 8, mi recall es 80%
# Recall alto = No se me escapan incendios (bueno para la seguridad)

# Precisión detecta si un modelo está exagerando el reconocimiento de clase positiva al asignar demasiadas etiquetas positivas.
# Precision: "¿Qué tan bueno soy evitando falsas alarmas?"
# Pregunta clave: De todas las veces que digo "¡INCENDIO!", ¿cuántas veces realmente hay fuego?
# Miedo principal: Dar demasiadas falsas alarmas (Falso Positivo)
# Ejemplo: Si doy 10 alarmas y solo 7 son incendios reales, mi precision es 70%
# Precision alto = Pocas falsas alarmas (bueno para no molestar)

# ### Recall alto:
# - "Mi modelo encuentra a casi todas las personas que van a reclamar"
# - Riesgo: Puede que también marque a muchas personas que NO van a reclamar

# ### Precision alto:
# - "Cuando mi modelo dice que alguien va a reclamar, casi siempre tiene razón"
# - Riesgo: Puede que se pierda algunas personas que sí van a reclamar

# Recall alto + Precision baja: Detector muy sensible (muchas falsas alarmas)
# Precision alta + Recall bajo: Detector muy conservador (se pierde casos reales)

import pandas as pd
from sklearn.metrics import precision_score,recall_score
target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
precision = precision_score(target,predictions)
recall = recall_score(target,predictions)
f1 =  2 * precision * recall / (precision + recall)
print('Recall:', recall)
print('Precisión:', precision)
print('Puntuación F1', f1)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score #funcion
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print(f1_score(target_valid,predicted_valid))

#--------------------------------Ajuste de peso de clase------------------------------------
# los modelos también buscan una puntuación más alta, por lo que tienden a centrarse en las observaciones de mayor importancia

# Los algoritmos de aprendizaje automático consideran que todas las observaciones del conjunto de entrenamiento tienen la misma ponderación de forma predeterminada. 
# Si necesitamos indicar que algunas observaciones son más importantes, asignamos un peso a la clase respectiva.

#regresion logistica
# argumento: class_weight. Por defecto, es None, es decir, las clases son equivalentes
# class_weight='balanced'
# class "0" weight = 1.0
# class "1" weight = N #La clase "0" aparece 9 veces más que la clase "1". Entonces N = 9.
# Clase "0" (mayoritaria): peso = 1.0
# Clase "1" (minoritaria): peso = 9.0
# Cada error en la clase "0" cuenta como 1 punto de penalización
# Cada error en la clase "1" cuenta como 9 puntos de penalización

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))

#arboles de decision
# argumento: class_weight. 

#--------------------------------Sobremuestro-----------------------------------
# Las tareas más importantes se repiten varias veces para que sean más fáciles de recordar.
# 1.Se divide el dataset de entrenamiento en observaciones negativas y positivas.
# 2.Se duplican aleatoriamente varias veces las observaciones positivas (las que raramente ocurren).
# 3.Se crea una nueva muestra de entrenamiento con base en los datos obtenidos.
# 4.Se mezclan los datos: hacer la misma pregunta una y otra vez no ayudará al entrenamiento.

# Situación original:
# - Tienes 100 preguntas fáciles (clase mayoritaria)
# - Tienes 10 preguntas difíciles (clase minoritaria)
# - Cada pregunta vale 1 punto

# Problema: Es muy fácil olvidar las preguntas difíciles porque son pocas.

# Solución con sobremuestreo:
# - Repites las 10 preguntas difíciles 10 veces cada una
# - Ahora tienes: 100 fáciles + 100 difíciles (copias)
# - ¡Las clases están balanceadas!

# No inventamos información nueva
# solo mostramos más veces los mismos patrones reales
# El modelo aprende mejor los patrones raros
# como si repitieras las preguntas de historia varias veces para estudiar mejor
# Mejora el balance
# ahora el modelo ve suficientes ejemplos de ambas clases

import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
features_zeros = features_train[target_train == 0] #"dame las filas donde target_train es 0"
features_ones = features_train[target_train == 1]
target_zeros  = target_train[target_train == 0]
target_ones  = target_train[target_train == 1]
print(features_zeros.shape)
print(features_ones.shape)
print(target_zeros.shape)
print(target_ones.shape)

#repetir los elementos de la lista - sintaxis
answers = [0, 1, 0]
answers_x3 = answers * 3
print(answers_x3)

repeat = 10
features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat) #concatena dataframes
target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
print(features_upsampled.shape)
print(target_upsampled.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle #mezcla aleatoriamente las filas de tus datos
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros  = features[target == 0]
    target_ones  = features[target == 1]
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    return features_upsampled, target_upsampled
features_upsampled, target_upsampled = upsample(
    features_train, target_train, 10
)
print(features_upsampled.shape)
print(target_upsampled.shape)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_upsampled, target_upsampled) #entrenado con datos sobremuestreados
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))

#--------------------------------Submuestro-----------------------------------
# técnica que se utiliza para equilibrar las clases dentro de un conjunto de datos de entrenamiento.
# reducir la frecuencia de las observaciones de una clase predominante, para mejorar el rendimiento del modelo al evitar que esté sesgado hacia la clase más común.

features_sample = features_train.sample(frac=0.1, random_state=12345) #la función seleccionará aleatoriamente el 10% de los elementos para formar una nueva muestra.
print(features_sample.shape)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
def downsample(features, target, fraction): #separar las clases 
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    features_downsampled = pd.concat( #particion y union
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat( #submuestreo
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )
    features_downsampled, target_downsampled = shuffle( #mezcla
        features_downsampled, target_downsampled, random_state=12345
    )
    return features_downsampled, target_downsampled
features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.1
)
print(features_downsampled.shape)
print(target_downsampled.shape)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_downsampled, target_downsampled) #entrenado con datos submuestro
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))

#--------------------------------Umbral de clasificación-----------------------------------
# la regresión logística calcula la probabilidad de cada clase.
# Dado que solo tenemos dos clases (cero y uno), la probabilidad de la clase "1" es la que nos interesa
# Esta probabilidad varía de cero a uno: mayor a 0.5, observación positiva; si es menor, negativa.

# El punto de corte entre clasificaciones positivas y negativas se llama umbral. Por defecto es 0.5
# La precisión disminuirá, recall aumentará

#---------------------------------Ajuste de umbral------------------------------------
# predict_proba(), proporciona la probabilidad de que cada observación pertenezca a cada clase posible.
# Esta toma las características de las observaciones como entrada y devuelve un array de probabilidades para cada clase.
# disponible  para árboles de decisión y bosques aleatorios

probabilities = model.predict_proba(features)
# Este modelo genera dos probabilidades para cinco observaciones. 
# [[0.5795 0.4205]
#  [0.6629 0.3371]
#  [0.7313 0.2687]
#  [0.6728 0.3272]
#  [0.5086 0.4914]]
#     0       1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1] #obtener solo las probabilidades de la clase "1"
print(probabilities_one_valid[:5]) #mostrar las primeras 5 probabilidades

#Queremos ver cómo cambian la precisión y recall cuando modificamos el umbral de decisión del modelo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid) #nos da probabilidades para ambas clases
probabilities_one_valid = probabilities_valid[:, 1]
for threshold in np.arange(0, 0.3, 0.02): #permitir un bucle con decimales
    predicted_valid = probabilities_one_valid > threshold #Compara cada probabilidad con el umbral
    precision = precision_score(target_valid,predicted_valid) #usa true o false comparado con el valid
    recall = recall_score(target_valid,predicted_valid)
    print(
        	'Threshold = {:.2f} | Precision = {:.3f}, Recall = {:.3f}'.format(
            	threshold, precision, recall
        	)
    	)

#resultado:
Threshold = 0.00 | Precision = 0.013, Recall = 1.000
Threshold = 0.02 | Precision = 0.052, Recall = 0.645
Threshold = 0.04 | Precision = 0.061, Recall = 0.609
Threshold = 0.06 | Precision = 0.072, Recall = 0.367
Threshold = 0.08 | Precision = 0.097, Recall = 0.254
Threshold = 0.10 | Precision = 0.112, Recall = 0.178
Threshold = 0.12 | Precision = 0.146, Recall = 0.107
Threshold = 0.14 | Precision = 0.033, Recall = 0.012
Threshold = 0.16 | Precision = 0.034, Recall = 0.006
Threshold = 0.18 | Precision = 0.000, Recall = 0.000
Threshold = 0.20 | Precision = 0.000, Recall = 0.000
Threshold = 0.22 | Precision = 0.000, Recall = 0.000
Threshold = 0.24 | Precision = 0.000, Recall = 0.000
Threshold = 0.26 | Precision = 0.000, Recall = 0.000
Threshold = 0.28 | Precision = 0.000, Recall = 0.000

# Umbral bajo (ej: 0.02): Predice "Sí reclamo" más frecuentemente
# Recall alto: Encuentra casi todos los reclamos reales
# Precisión baja: También predice muchos falsos positivos

# Umbral alto (ej: 0.28): Predice "Sí reclamo" menos frecuentemente
# Precisión alta: Cuando predice reclamo, suele acertar
# Recall bajo: Se pierde muchos reclamos reales

#---------------------------------TVP y TFP------------------------------------
TVP = VP / P
# TVP(Tasa de verdaderos positivos) es el resultado de las respuestas VP divididas entre todas las respuestas positivas.

TFP = FP / N
# TFP(Tasa de falsos positivos) es el resultado de las respuestas FP divididas entre todas las respuestas negativas.

# los denominadores son valores fijos que no cambian con ajustes en el modelo, por lo que no existe riesgo de división entre cero.

#---------------------------------Curva ROC------------------------------------
# la curva ROC es una línea diagonal que va desde la esquina inferior izquierda hasta la esquina superior derecha
# Cuanto más se aleje la curva ROC de esta línea diagonal hacia la esquina superior izquierda, mejor será el modelo

# Para encontrar cuánto difiere nuestro modelo del modelo aleatorio, calculemos el valor AUC-ROC (Área Bajo la Curva ROC).

# AUC-ROC para un modelo aleatorio es 0.5.

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(target, probabilities) #valores TFP, valores TVP y los umbrales que superó.
# ambas usan predict_proba()

import pandas as pd #graficar la curva ROC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid) 
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--') #dibuja una línea diagonal punteada que va desde el punto (0,0) hasta el punto (1,1)(modelo aleatorio)
plt.xlim([0.0, 1.0]) #establecer el límite para los ejes de 0 a 1
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.show()

# ¿Qué es AUC-ROC?
# Es un número único que resume qué tan buena es tu curva ROC.
# No depende del umbral (evalúa todos los umbrales posibles)

# Predicciones binarias:
# - Son valores definitivos: 0 o 1
# - Se obtienen con model.predict()
# - Ejemplo: [0, 1, 0, 1, 1]

# Probabilidades:
# - Son valores entre 0 y 1 que indican la "confianza" del modelo
# - Se obtienen con model.predict_proba()
# - Ejemplo: [0.2, 0.8, 0.3, 0.9, 0.7]

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')
target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print(auc_roc)

# 0.5: El modelo no es mejor que adivinar al azar (línea diagonal en la curva ROC)
# 0.5 - 0.7: Rendimiento pobre a aceptable
# 0.7 - 0.8: Rendimiento aceptable a bueno
# 0.8 - 0.9: Rendimiento bueno a excelente
# 0.9 - 1.0: Rendimiento excelente (pero cuidado con el sobreajuste)

#---------------------------------Preparacion de datos------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('/datasets/flights.csv')
data_ohe = pd.get_dummies(data, drop_first=True) #Convertiste las variables categóricas en numéricas usando One-Hot Encoding. El parámetro drop_first=True evita la multicolinealidad
target = data_ohe['Arrival Delay']
features = data_ohe.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
numeric = ['Day', 'Day Of Week', 'Origin Airport Delay Rate',
       'Destination Airport Delay Rate', 'Scheduled Time', 'Distance',
       'Scheduled Departure Hour', 'Scheduled Departure Minute']
scaler = StandardScaler() #Estandarizaste solo las variables numéricas para que tengan media 0 y desviación estándar 1.
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
print(features_train.shape)
print(features_valid.shape)

#---------------------------------EMC en una tarea dew regresion------------------------------------
# MSE: Mean Squared Error
# Es el promedio de los errores al cuadrado entre las predicciones y los valores reales.
# Fórmula:
# MSE = (1/n) × Σ(valor_real - predicción)²
# Ejemplo con retrasos de vuelos:
# Retraso real: 30 minutos
# Predicción: 25 minutos
# Error: 30 - 25 = 5 minutos
# Error al cuadrado: 5² = 25

# RMSE: Root Mean Squared Error
# Es la raíz cuadrada del MSE.
# Fórmula:
# RMSE = √MSE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/datasets/flights_preprocessed.csv')
target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LinearRegression()
model.fit(features_train,target_train)
predicted_valid = model.predict(features_valid)  #Toma las características de los vuelos de validación y predice cuántos minutos de retraso tendrá cada vuelo.
mse = mean_squared_error(target_valid,predicted_valid) #Compara las predicciones con los retrasos reales y calcula qué tan lejos estuvieron.
print("MSE =", mse) #Significa que, en promedio, tus predicciones se desvían del valor real por aproximadamente x unidades al cuadrado.
print("RMSE =", mse ** 0.5) #Este valor está en las mismas unidades que tu variable objetivo, en promedio, tus predicciones se desvían por aproximadamente x minutos del valor real.

predicted_valid = pd.Series(target_train.mean(), index=target_valid.index) #Crea una serie con esa media repetida para cada observación de validación
mse = mean_squared_error(target_valid, predicted_valid)
print('Mean')
print('MSE =', mse)
print('RMSE =', mse ** 0.5)

# ¿Qué es un modelo constante?
# modelo simple: siempre predice el mismo valor para todas las observaciones.
# En este caso, predice la media del conjunto de entrenamiento.
# Ejemplo:
target_train = [10, 20, 30, 15, 25]  # Media = (10+20+30+15+25)/5 = 20 minutos
# El modelo constante predice SIEMPRE 20 minutos:
predicted_valid = [20, 20, 20, 20, 20]  # Para TODOS los vuelos de validación

# Modelo Constante:
# Vuelo corto de día → Predice: 25 minutos
# Vuelo largo de noche → Predice: 25 minutos
# Vuelo en tormenta → Predice: 25 minutos

# Regresión Lineal:
# Vuelo corto de día → Predice: 15 minutos
# Vuelo largo de noche → Predice: 35 minutos
# Vuelo en tormenta → Predice: 55 minutos

# La regresión lineal se equivoca en promedio por 46.15 minutos
# El modelo constante se equivoca en promedio por 48.57 minutos
# ¡La regresión lineal es 2.4 minutos más precisa!

#---------------------------------predict vs predict_proba------------------------------------
# Para regresión 
predicted = model.predict(features_valid)
# Resultado: [25.3, 15.7, 42.1, ...]  # Minutos de retraso predichos
# Para clasificación:
predicted = model.predict(features_valid)
# Resultado: [0, 1, 0, 1, ...]  # Clases predichas directamente

# Solo funciona para clasificación:
probabilities = model.predict_proba(features_valid)
# Resultado: [[0.8, 0.2], [0.3, 0.7], ...]  # [prob_clase_0, prob_clase_1]
# NO funciona para regresión 

#---------------------------------Coeficiente de determinacion-----------------------------------
# Los valores absolutos son medidas que se expresan en las mismas unidades que los datos originales
# Los valores relativos son medidas que se expresan como proporciones o porcentajes, sin unidades específicas.

# R² (Coeficiente de determinación): Va de 0 a 1 (o puede ser negativo si el modelo es muy malo)
# R² = 0.8 significa que el modelo explica el 80% de la variabilidad de los datos
# R² = 0 significa que el modelo funciona igual que usar solo la media
# R² = 1 sería un modelo perfecto (ECM del modelo es cero)
# Cuando R2 es negativo, peor que usar solo la media para predecir.

# divide el  ECM del modelo entre el ECM de la media y luego resta el valor obtenido de uno

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score #Coeficiente de determinacion
data = pd.read_csv('/datasets/flights_preprocessed.csv')
target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print('R2 =', r2_score(target_valid,predicted_valid))

#---------------------------------Maximacion del r2-----------------------------------
# R² te dice qué tan bueno es tu modelo comparado con usar solo la media

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("/datasets/flights_preprocessed.csv")
target = data["Arrival Delay"]
features = data.drop(["Arrival Delay"], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
for depth in range(1, 16, 1):
    model = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    
    r2_train = model.score(features_train, target_train)
    r2_valid = model.score(features_valid, target_valid)
    mae_valid = mean_absolute_error(target_valid, model.predict(features_valid))
    
    print(f"Depth {depth}: R2 valid = {r2_valid:.4f}, MAE = {mae_valid:.2f}, R2 train = {r2_train:.2f}")

# Si R2 de entrenamiento >> R2 de validación: Sobreajuste (el modelo memorizó pero no generaliza)
# Si ambos valores son similares: Buen balance

#---------------------------------Error absoluto medio-----------------------------------
# Mean Absolute Error
# Mide qué tan lejos están, en promedio, tus predicciones de los valores reales.

target =      [-0.5, 2.1, 1.5, 0.3]  # Valores reales
predictions = [-0.6, 1.7, 1.6, 0.2]  # Predicciones del modelo
# Paso 1: Calcula la diferencia absoluta para cada par:
# |(-0.5) - (-0.6)| = |0.1| = 0.1
# |2.1 - 1.7| = |0.4| = 0.4
# |1.5 - 1.6| = |-0.1| = 0.1
# |0.3 - 0.2| = |0.1| = 0.1
# Paso 2: Suma todos los errores:
0.1 + 0.4 + 0.1 + 0.1 = 0.7
# Paso 3: Divide entre el número de observaciones:
0.7 / 4 = 0.175 #"Mi modelo se equivoca en promedio 0.175 unidades"

import pandas as pd
def mae(target, predictions):
    error = 0 #Inicializa una variable para acumular la suma de errores absolutos
    for i in range(target.shape[0]): #series empieza en 0 hasta n-1
            # target.shape[0] = 4 (porque tienes 4 valores)
            # El bucle ejecutará: i = 0, 1, 2, 3
        error += abs(target[i] - predictions[i]) #Resta predicción de valor real y toma el valor absoluto
    return error / target.shape[0] #Divide la suma total de errores entre el número de observaciones
target = pd.Series([-0.5, 2.1, 1.5, 0.3])
predictions = pd.Series([-0.6, 1.7, 1.6, 0.2])
print(mae(target, predictions))

# n tu caso, tus predicciones se desvían en promedio 0.175 unidades de los valores verdaderos.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('/datasets/flights_preprocessed.csv')
target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print(mean_absolute_error(target_valid,predicted_valid)) # Hace exactamente lo mismo, pero está optimizada y es la función estándar de la industria

#---------------------------------Interpretacion de EAM (MAE)-----------------------------------
Real:        [10, 20, 30]
Predicción:  [12, 18, 35]
Errores:     [2,  2,  5]  # valores absolutos
EAM = (2 + 2 + 5) / 3 = 3
Interpretación = "Mi modelo se equivoca en promedio 3 unidades"

# Más fácil de interpretar (mismas unidades que los datos)
# Menos sensible a valores extremos
# Trata todos los errores por igual

# Ejemplo: Prediciendo tiempo de llegada de un taxi
# Error de 5 min = molesto
# Error de 10 min = muy molesto
# Error de 20 min = súper molesto
# Pero: ¿20 min es realmente 4 veces peor que 10 min? Tal vez no tanto.

# EAM: "Quiero saber el error típico promedio"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('/datasets/flights_preprocessed.csv')
target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print('Linear Regression')
print(mean_absolute_error(target_valid, predicted_valid))
print()
predicted_valid = pd.Series(target_train.median(), index=target_valid.index)
print('Median')
print(mean_absolute_error(target_valid, predicted_valid))

# 1. Ambos modelos se equivocan aproximadamente 27 minutos en promedio
# Si un vuelo llega con 10 min de retraso, tu modelo podría predecir 37 min o -17 min
# Es un error bastante grande para retrasos de vuelos

# 2. La mediana ganó (¡apenas!)
# Diferencia: solo 0.22 minutos (13 segundos)
# El modelo "tonto" superó al "inteligente"

#---------------------------------Minimizacion de EAM (MAE)-----------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
data = pd.read_csv("/datasets/flights_preprocessed.csv")
target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)
model = RandomForestRegressor(n_estimators=70, max_depth=10, random_state=12345)
model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)
print("Configuración del modelo actual lograda:")
print(
    "Valor EAM en un conjunto de entrenamiento: ",
    mean_absolute_error(target_train, predictions_train),
)
print(
    "Valor EAM en un conjunto de validación: ",
    mean_absolute_error(target_valid, predictions_valid),
)

# 1. Regresión Lineal vs Random Forest
# Regresión Lineal: Solo puede encontrar relaciones lineales (líneas rectas)
# Regresión Lineal: "Si llueve un poco más, el retraso aumenta proporcionalmente"

# Random Forest: Puede capturar relaciones complejas y no lineales
# Random Forest: "Si llueve poco, no pasa nada. Si llueve mucho, los retrasos se disparan. Si es fin de semana Y llueve, es aún peor"

#---------------------------------ECM (MSE)-----------------------------------
Real:        [10, 20, 30]
Predicción:  [12, 18, 35]
Errores:    [4,  4,  25]  # errores elevados al cuadrado
ECM = (4 + 4 + 25) / 3 = 11

# Penaliza más los errores grandes (por el cuadrado)
# Más sensible a outliers
# Unidades al cuadrado (más difícil de interpretar)

# Ejemplo: Prediciendo temperatura de un reactor nuclear
# Error de 1°C = ok
# Error de 5°C = preocupante
# Error de 10°C = ¡CATÁSTROFE!
# Aquí: 10°C de error SÍ es muchísimo peor que 5°C. Quieres penalizar duramente los errores grandes.
# ECM: "Los errores grandes me preocupan MUCHO más que los pequeños"

#---------------------------------Hechos, emociones y valoraciones-----------------------------------
# "La plataforma en línea está congelada" es un hecho
# Un hecho es algo que es objetivamente cierto.

# "Me molesta" es una emoción
# Una emoción es una respuesta subjetiva.  Un solo suceso (también un pensamiento) puede causar diferentes emociones

# "Estúpida plataforma en línea" es una valoración
# Una valoración es una opinión sobre un suceso, un tema o una persona, expresada de forma emotiva. 

# trabaja principalmente con hechos
# evita expresar emociones cuando las personas esperan una opinión experta de ti
# expón tus valoraciones y juicios, y aclara que es tu opinión personal
# Que sus conclusiones se tengan o no en cuenta depende de la claridad de la redacción y de la lógica de la justificación.




























































































































































































#---------------------------------Curva PR------------------------------------
# En la gráfica, el valor de precisión se traza verticalmente y recall, horizontalmente. 
# Una curva trazada a partir de los valores de Precisión y Recall se denomina curva PR. 
# Cuanto más alta sea la curva, mejor será el modelo.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
precision, recall, thresholds = precision_recall_curve(
    target_valid, probabilities_valid[:, 1]
)

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()















































































































































































