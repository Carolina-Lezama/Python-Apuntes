#---------------------------- Series temporales ----------------------------
# ¿Qué es una serie temporal?
# Son secuencias de números [0,1,2,3,...] a lo largo del eje del tiempo
# Ejemplos(Variables que cambian a lo largo del tiempo):
#     - El precio de las acciones de una empresa al comienzo del negocio.
#     - El volumen de ventas de una tienda online por día.
#     - El número de jugadores en línea por hora.

# Generalmente, el intervalo entre los valores de la serie es constante: un día, una hora o 10 minutos.

#---------------------------- Ejemplo para trabajar con fechas y horas en pandas ----------------------------
data = pd.read_csv('/datasets/energy_consumption.csv')
print(data.head())

#Resultado:
#               Datetime  PJME_MW
# 0  2002-12-31 01:00:00  26498.0
# 1  2002-12-31 02:00:00  25147.0
# 2  2002-12-31 03:00:00  24574.0
# 3  2002-12-31 04:00:00  24393.0
# 4  2002-12-31 05:00:00  24860.0

# Datetime: El valor de fecha y hora a lo largo del eje de tiempo. 
#           En esta tabla, el intervalo es de una hora.

# PJME_MW: consumo de electricidad por hora.

#---------------------------- Pasar de object a datatime ----------------------------
data = pd.read_csv('/datasets/energy_consumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
print(data.info())

# Resultado
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 145366 entries, 0 to 145365
# Data columns (total 2 columns):
#  #   Column    Non-Null Count   Dtype
# ---  ------    --------------   -----
#  0   Datetime  145366 non-null  datetime64[ns]
#  1   PJME_MW   145366 non-null  float64
# dtypes: datetime64[ns](1), float64(1)
# memory usage: 2.2 MB

#---------------------------- Establecer las fechas como index ----------------------------
data = pd.read_csv('/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0])
print(data.info())

#parse_dates=[0] -> pandas convierte la primera columna (índice 0) en un objeto datetime.
#index_col=[0]: Usa la columna 0 (Datetime) como índice

# Resultado
# <class 'pandas.core.frame.DataFrame'>
# DatetimeIndex: 145366 entries, 2002-12-31 01:00:00 to 2018-01-02 00:00:00
# Data columns (total 1 columns):
#  #   Column   Non-Null Count   Dtype
# ---  ------   --------------   -----
#  0   PJME_MW  145366 non-null  float64
# dtypes: float64(1)
# memory usage: 2.2 MB

#---------------------------- Otras formas de usar parse_dates y index_col ----------------------------
# Por número de columna
parse_dates=[0, 2]  # Columnas 0 y 2
# Por nombre de columna  
parse_dates=['Datetime', 'Created_at']

# Por posición (número)
index_col=[0]  # Primera columna (posición 0)
# Por nombre (texto)
index_col=['Datetime']  # Columna llamada 'Datetime'

#---------------------------- Ordenar fechas cronologicamente ----------------------------
data.sort_index(inplace=True)
print(data.index.is_monotonic_increasing) #verificar si las fechas y horas están en orden cronológico, da boolen
print(data.info())

#---------------------------- Seleccionar rangos de fechas ----------------------------
data = data['2018-01': '2018-06'] #Ejemplos
data_1 = data['2016': '2017'] 
print(data.info())

#---------------------------- Trazar el gráfico de series temporales. ----------------------------
data.plot()

#---------------------------- Remuestreo - Remuestrar ----------------------------
#Significa cambiar el intervalo con los valores de la serie. 
# 1. Elige la nueva duración del intervalo: 
#     los valores del intervalo existente están agrupados, la nueva agrupación se basa en la nueva duración del intervalo, sumar o dividir los intervalos existentes
# 2. Elige la función de agregación:
#     Cada grupo se calcula el valor acumulado de la serie(mediana, media, máximo o mínimo)

#Cambiar el intervalo y agrupar los valores
# 1H = una hora
data.resample('1H') 
# 2W = dos semanas
data.resample('2W')

#.resample() es similar a la función .groupby(), después de agrupar, llama alguna funcion
# media para cada hora
data.resample('1H').mean()
# máximo por cada dos semanas
data.resample('2W').max()

#Ejemplo: Grafico de promedio por año
data.sort_index(inplace=True)
data = data.resample('1Y').mean()
data.plot()

#Ejemplo: Grafico de consumo total por día entre enero y junio de 2018
data.sort_index(inplace=True)
data = data['2018-01': '2018-06'].resample('1D').sum()
data.plot()

#---------------------------- Media móvil para reducir las fluctuaciones en una serie temporal ----------------------------
# ¿Que es la media móvil o promedio móvil?
# método para suavizar los datos en una serie temporal para ver la tendencia real
#     Cuanto mayor sea el intervalo, más fuerte será el suavizado. 
#     En cada punto se calcula el valor medio(desde el principio hasta el final de la serie temporal).
# consiste en encontrar los valores menos susceptibles a fluctuaciones, es decir, la media aritmética.
# el número de medias obtenidas será ligeramente menor que el número de valores iniciales de la serie

# ¿Qué es una fluctuación?
# Una fluctuación es cuando los valores suben y bajan de forma irregular, como las olas del mar.

# En pandas:
# 1. crear una ventana móvil. Especifica el tamaño de la ventana en el argumento
# 2. Agregar la media móvil a cada ventana
data.rolling(7).mean()

#Ejemplo:
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
data['rolling_mean'] = data.rolling(10).mean()
data.plot()

#---------------------------- Tendencia  ----------------------------
# Es un cambio ligero del valor medio de la serie sin repetir patrones(Variable). 
# Es como la dirección general hacia donde se mueven los datos a lo largo del tiempo.

# Por ejemplo:
#     el incremento anual en la venta de boletos de avión.
#     la temperatura aumenta desde la mañana hasta la tarde, y luego disminuye. 

# Es el patrón de largo plazo que muestra si los valores están aumentando, disminuyendo o manteniéndose estables.
# Ignoras las pequeñas fluctuaciones y te enfocas en la dirección general.

#---------------------------- Estacionalidad ----------------------------
# Patrones que se repiten de forma cíclica en una serie temporal (como mayor consumo en ciertos periodos)
# Por ejemplo:
    # el crecimiento de las ventas de boletos de avión cada verano.

#---------------------------- Aplicacion y precauciones  ----------------------------
# Las tendencias y la estacionalidad dependen de la escala de los datos. 
# No puedes ver patrones que se repiten todos los veranos si solo hay datos de un año.

from statsmodels.tsa.seasonal import seasonal_decompose
#toma una serie temporal y devuelve una instancia de la clase DecomposeResult
    # decomposed.trend — tendencia
    # decomposed.seasonal — componente estacional
    # decomposed.resid — residuos
decomposed = seasonal_decompose(data) #divide la serie en tres componentes: tendencia, estacionalidad y residuos.
plt.figure(figsize=(8, 8))
plt.subplot(311)
    # 3 = Número total de filas (3 gráficos uno encima del otro)
    # 1 = Número total de columnas (1 sola columna)  
    # 1 = Posición del gráfico actual (el primero)
decomposed.trend.plot(ax=plt.gca()) #seleccionar un grafico especifico para usar
plt.title('Tendencia')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Estacionalidad')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuales')
plt.tight_layout()    # esto ayuda a encajar gráficos secundarios en el área

# Para obtener la gráfica inicial, necesitamos sumar los tres juntos. Cada gráfico tiene sus propias unidades
print(decomposed.trend[4:6] + decomposed.seasonal[4:6] + decomposed.resid[4:6])
print()
print(df[4:6])

# Resultado
# date
# 2020-09-03     9.0
# 2020-09-04    11.0
# dtype: float64

#             value
# date             
# 2020-09-03      9
# 2020-09-04     11

#Ejemplo:
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
decomposed = seasonal_decompose(data)
decomposed.seasonal['2018-01-01':'2018-01-15'].plot() #Traza un gráfico para el componente estacional de los primeros 15 días de enero de 2018.

#---------------------------- Residuos ----------------------------
# Lo que queda después de quitar tendencia y estacionalidad
# es esencialmente solo ruido.

#---------------------------- Series estacionarias ----------------------------
# pueden ayudar a pronosticar datos

#desviación estándar móvil y promedio estándar móvil
import pandas as pd
data = pd.read_csv('energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
data['mean'] = data['PJME_MW'].rolling(15).mean()
data['std'] = data['PJME_MW'].rolling(15).std()
data.plot()

#---------------------------- Series estacionarias en estadistica ----------------------------
# se describe como un proceso estocástico. 
#     Estocástico significa "aleatorio" o "impredecible": cada resultado es aleatorio, pero sigue ciertas reglas.
# Un proceso estocástico es estacionario si su distribución no cambia con el tiempo. 
# Si la distribución cambia, entonces el proceso estocástico es no estacionario.

# la serie temporal estacionaria es una serie en la que la media y la desviación estándar no cambian.
# Las series de tiempo no estacionarias son más difíciles de pronosticar porque sus propiedades cambian demasiado rápido.

# Tiene variación aleatoria y su distribución cambia con el tiempo. 
# Tiene una media y una varianza y estos valores también cambian.

# Un proceso estocástico es una serie de datos que tiene:
#     Variación aleatoria (no puedes predecir exactamente el siguiente valor)
#     Patrones estadísticos (pero sí puedes conocer la tendencia general)

#---------------------------- ¿que es la desviacion estandar? ---------------------------
# mide qué tan dispersos están tus datos respecto al promedio.
# Valores estan cerca o lejos de la media

#  Desviación estándar BAJA:
# - Datos consistentes
# - Poca variabilidad
# - Más predecible

#  Desviación estándar ALTA:
# - Datos variables
# - Mucha fluctuación
# - Menos predecible

#---------------------------- hacer estacionaria a una serie ----------------------------
# tomar las diferencias de sus valores

#---------------------------- Diferencias de series temporales ----------------------------
# es una secuencia de diferencias entre elementos vecinos de una serie temporal (es decir, el valor anterior se resta del siguiente)

# ¿Qué son las "diferencias entre elementos vecinos"?
# restar cada valor con el valor anterior. Como comparar "¿cuánto cambió de ayer a hoy?"

### Serie original (consumo eléctrico):
# Día 1: 150 kWh
# Día 2: 160 kWh  
# Día 3: 140 kWh
# Día 4: 170 kWh

### Diferencias (cambios día a día):
# Día 1: ??? (no hay día anterior)
# Día 2: 160 - 150 = +10 kWh (subió 10)
# Día 3: 140 - 160 = -20 kWh (bajó 20)  
# Día 4: 170 - 140 = +30 kWh (subió 30)

### En lugar de preguntar:
# - "¿Cuánto consumí hoy?" (valor absoluto)
### Preguntas:
# - "¿Cuánto cambió mi consumo respecto a ayer?" (diferencia)

data.shift() #se usa para encontrar las diferencias de series temporales
# Todos los valores se desplazan un paso hacia adelante a lo largo del eje de tiempo

data = pd.Series([0.5, 0.7, 2.4, 3.2])
print(data)
print(data.shift(fill_value=0))

#Resultado:
# 0    0.5
# 1    0.7
# 2    2.4
# 3    3.2
# dtype: float64
# 0    NaN - 0 #depende si relleno o no
# 1    0.5
# 2    0.7
# 3    2.4
# dtype: float64

print(data-data.shift()) #encontrar las diferencias
#Resultado:
# 0    NaN
# 1    0.2
# 2    1.7
# 3    0.8
# dtype: float64

# Las diferencias de las series temporales son más estacionarias que la serie en sí.

#Ejemplo:
import pandas as pd
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data['2018-01':'2018-06'].resample('1D').sum()
data -= data.shift()
data['mean'] = data['PJME_MW'].rolling(15).mean()
data['std'] = data['PJME_MW'].rolling(15).std()
data.plot()

#---------------------------- objetivo del pronóstico de series temporales ----------------------------
# desarrollar un modelo que prediga los valores futuros de una serie temporal con base en datos anteriores.
# El periodo en el futuro para el que se prepara el pronóstico se conoce como horizonte de pronóstico.

# Si los valores de una serie temporal o de la función x(t) (donde t = tiempo) 
#     Si son números: te enfrentas a una tarea de regresión para la serie temporal. 
#     Si son categorías, será una tarea de clasificación.

# ejemplos de pronósticos de series temporales:
#     el número de pedidos de taxi para la próxima hora.
#     el número de entregas del servicio de mensajería para el día siguiente.
#     el nivel de demanda de productos en una tienda en línea la próxima semana.

# No puedes mezclar los conjuntos en el pronóstico de series temporales(entramiento-prueba) porque el modelo podría aprender de los datos futuros, lo que se llama "fuga de datos".
# los datos del conjunto de entrenamiento deben preceder a los datos del conjunto de prueba.
#     En series temporales, el tiempo es sagrado. No puedes usar datos del futuro para predecir el pasado.
#     el modelo no debe entrenarse con datos futuros.

# ¿Qué NO debes hacer?
### División incorrecta ( "usar el futuro para predecir el pasado" ):
# Datos originales: Ene, Feb, Mar, Abr, May, Jun
# ❌ Entrenamiento: Ene, Mar, May, Jun
# ❌ Prueba: Feb, Abr

# ¿Qué SÍ debes hacer?
### División correcta (secuencial):
# Datos originales: Ene, Feb, Mar, Abr, May, Jun
# ✅ Entrenamiento: Ene, Feb, Mar, Abr (80%)
# ✅ Prueba: May, Jun (20%)

import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.Series([0.1, 0.5, 2.3, 1.2, 1.5])
train, test = train_test_split(data, shuffle=False, test_size=0.2) # que no se mezclen los datos de forma predeterminada
print('Conjunto de entrenamiento:')
print(train)
print('Conjunto de prueba:')
print(test)

# Resultado:
# Conjunto de entrenamiento:
# 0    0.1
# 1    0.5
# 2    2.3
# 3    1.2
# dtype: float64
# Conjunto de prueba:
# 4    1.5
# dtype: float64

import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data.resample('1D').sum()
train, test = train_test_split(data, shuffle=False, test_size=0.2)
print(train.index.min(), train.index.max()) #valores mínimo y máximo de los índices 
print(test.index.min(), test.index.max())

#---------------------------- entrenar un modelo de linea base con un horizonte de pronóstico de un día usando la mediana ----------------------------
# útiles para automatizar la toma de decisiones técnicas.

# pronosticar series temporales sin entrenamiento:
#     1.Todos los valores de la muestra de prueba se pronostican con el mismo número (una constante).
#     utilizando la mediana (también se podrían utilizar otras medidas como la media aritmética)
#     2. El nuevo valor x(t) se predice mediante el valor anterior de la serie, definido como x(t-1)
#     es independiente de la métrica, el último valor de los datos de entrenamiento se puede utilizar como el primer valor de los datos de prueba

# ejemplo: 1. pronóstico de la mediana
### Tu estrategia fue:
# - "Todos los días futuros tendrán el mismo consumo"
# - Ese consumo = mediana de los datos históricos
### Es como decir:
# - "Mañana consumiré lo mismo que el día típico del pasado"

# ¿Qué mediste?
### EAM (Error Absoluto Medio):
# - Mide qué tan lejos están tus predicciones de la realidad
# - EAM bajo = predicciones más exactas
# - EAM alto = predicciones menos exactas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data.resample('1D').sum()
train, test = train_test_split(data, shuffle=False, test_size=0.2)
print('Consumo medio diario de energía:', test['PJME_MW'].median())
pred_median = np.ones(test.shape) * train['PJME_MW'].median() #crea un array de predicciones con el mismo número (la mediana) para cada valor en el conjunto de prueba
print('EAM:', mean_absolute_error(test, pred_median))

# ejemplo: 2. utilizando el valor anterior de la serie. 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data.resample('1D').sum()
train, test = train_test_split(data, shuffle=False, test_size=0.2)
print('Consumo medio diario de energía:', test['PJME_MW'].median())
pred_previous = test.shift()
pred_previous.iloc[0] = train.iloc[-1]
print('EAM:', mean_absolute_error(test, pred_previous))

#---------------------------- crear características para un horizonte de pronóstico de un paso ----------------------------
# 1. Características del calendario:
#     Por lo general, las tendencias y la estacionalidad se vinculan con una fecha específica

#lo que resta es presentarlo como columnas separadas
data = pd.read_csv('energy_consumption.csv', index_col=[0], parse_dates=[0])
data.sort_index(inplace=True)
data = data.resample('1D').sum()
data['año'] = data.index.year #característica que contiene años
data['díadelasemana'] = data.index.dayofweek #característica que contiene días de la semana
print(data.head(10))

# Resultado:
#              PJME_MW  year  dayofweek
# Datetime                             
# 2002-01-01  714857.0  2002          1
# 2002-01-02  822277.0  2002          2
# 2002-01-03  828285.0  2002          3
# 2002-01-04  809171.0  2002          4
# 2002-01-05  729723.0  2002          5
# 2002-01-06  727766.0  2002          6
# 2002-01-07  800012.0  2002          0
# 2002-01-08  824710.0  2002          1
# 2002-01-09  810628.0  2002          2
# 2002-01-10  755317.0  2002          3

# 2. Características de desfase
#     Los valores anteriores en la serie temporal te dirán si la función x(t) aumentará o disminuirá
#     usar la función shift() para obtener los valores de desfase

data['lag_1'] = data['PJME_MW'].shift(1) #mueve todos los valores una posición hacia abajo, dia anterior
data['lag_2'] = data['PJME_MW'].shift(2) #mueve los valores dos posiciones hacia abajo, dos días anteriores
data['lag_3'] = data['PJME_MW'].shift(3) #mueve los valores tres posiciones hacia abajo, tres días anteriores   
print(data.head(10))

# resultado:
#             PJME_MW     lag_1     lag_2     lag_3
# Datetime                                          
# 2002-01-01  714857.0       NaN       NaN       NaN
# 2002-01-02  822277.0  714857.0       NaN       NaN
# 2002-01-03  828285.0  822277.0  714857.0       NaN
# 2002-01-04  809171.0  828285.0  822277.0  714857.0

# No todos los valores de desfase están disponibles para las primeras fechas, por lo que estas líneas contienen NaN

# ¿Qué son las características de desfase (lag features)?
# nos permiten usar valores anteriores de nuestra serie temporal para predecir valores futuros. 
# Es como decir: "Para predecir el consumo de energía de hoy, necesito saber cuánto se consumió ayer, anteayer, etc."

# 3. Media móvil:
#     establece la tendencia general de la serie temporal.

data['rolling_mean'] = data['PJME_MW'].rolling(5).mean()
print(data.head(10))

# Resultado:
#              PJME_MW  rolling_mean
# Datetime                          
# 2002-01-01  714857.0           NaN
# 2002-01-02  822277.0           NaN
# 2002-01-03  828285.0           NaN
# 2002-01-04  809171.0           NaN
# 2002-01-05  729723.0      780862.6
# 2002-01-06  727766.0      783444.4

# La media móvil en t considera el valor actual de la serie x(t), el cual no queremos porque el valor actual pertenece al objetivo y no a las características
# asegúrate de usar características de desfase para excluir el valor actual cuando diseñes las características

# No puedes usar el valor que quieres predecir como característica para predecirlo.

# INCORRECTO
# Día 3: 100 kWh
# Día 4: 120 kWh  
# Día 5: 110 kWh ← ¡Este es tu OBJETIVO!, haces trampa
# Media móvil = (100 + 120 + 110) / 3 = 110

# CORRECTO
# Día 3: 100 kWh
# Día 4: 120 kWh  
# Día 5: ??? ← Esto es lo que quieres predecir
# Media móvil = (100 + 120) / 2 = 110

# ¿Por qué es importante?
### En la vida real:
# - No conoces el consumo de hoy hasta que termine el día
# - Solo tienes datos históricos (días anteriores)
# - Debes predecir sin conocer el valor actual

# Incorrecto: usar la respuesta para encontrar la respuesta
# Correcto: usar solo información que ya conoces

data['rolling_mean'] = data['PJME_MW'].shift().rolling(5).mean() #usa solo valores anteriores
data.shift() #toma los valores y los desplaza un período hacia adelante
data.rolling(5) #"ventana móvil" de 5 períodos, 5 valores consecutivos a la vez.

import pandas as pd
import numpy as np
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data.resample('1D').sum()
def make_features(data, max_lag, rolling_mean_size): #obtener partes de la fecha y crear características de desfase
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    #Crear características de desfase (lag features)
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)
    data['rolling_mean'] = data['PJME_MW'].shift().rolling(rolling_mean_size).mean() 
make_features(data, 4)
print(data.head())

# Resultado
#              PJME_MW  year  month  day  ...     lag_1     lag_2     lag_3     lag_4   rolling_mean
# Datetime                                ...
# 2002-01-01  714857.0  2002      1    1  ...       NaN       NaN       NaN       NaN   NaN
# 2002-01-02  822277.0  2002      1    2  ...  714857.0       NaN       NaN       NaN   NaN
# 2002-01-03  828285.0  2002      1    3  ...  822277.0  714857.0       NaN       NaN   NaN
# 2002-01-04  809171.0  2002      1    4  ...  828285.0  822277.0  714857.0       NaN   NaN
# 2002-01-05  729723.0  2002      1    5  ...  809171.0  828285.0  822277.0  714857.0   793647.5
# [5 rows x 10 columns]

#---------------------------- entrenamiento de un modelo de regresión lineal con las nuevas características ----------------------------
# 1. dividir los datos en conjuntos de entrenamiento y prueba
# 2. obtener las características de prueba del conjunto de entrenamiento
#     Los valores restantes y la media móvil se pueden calcular a partir de datos anteriores
#     Las características de los primeros valores del conjunto de prueba se encuentran al final del conjunto de entrenamiento.
#     Es imposible obtener las características para los primeros valores del conjunto de entrenamiento porque no hay datos previos con los cuales trabajar(se deben eliminar). 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv(
    '/datasets/energy_consumption.csv', index_col=[0], parse_dates=[0]
)
data.sort_index(inplace=True)
data = data.resample('1D').sum()
def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    
    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['PJME_MW'].shift(lag)
    data['rolling_mean'] = (
        data['PJME_MW'].shift().rolling(rolling_mean_size).mean()
    )
# elegimos valores de argumento al azar
make_features(data, 1, 1)
train, test = train_test_split(data, shuffle=False, test_size=0.2)
train = train.dropna() #Eliminar las filas con valores ausentes del conjunto de entrenamiento.
print(train.shape)
print(test.shape)
features_train = train.drop(['PJME_MW'], axis=1) #declarar x y y
target_train = train['PJME_MW']
features_test = test.drop(['PJME_MW'], axis=1)
target_test = test['PJME_MW']
model = LinearRegression()
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(
    'EAM para el conjunto de entrenamiento:', mean_absolute_error(target_train, pred_train)
)
print('EAM para el conjunto de prueba:', mean_absolute_error(target_test, pred_test))
