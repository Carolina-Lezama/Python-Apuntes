#---------------------------- Visión artificial ----------------------------
# Es un campo de la inteligencia artificial que intenta imitar la visión humana
# Enseñar a las máquinas a comprender imágenes y videos

#---------------------------- Tareas con redes neuronales ----------------------------
# La regresión lineal, el bosque aleatorio y la potenciación del gradiente son buenos para los datos tabulares:
#     las filas representan observaciones y las columnas almacenan características

# ¿Qué pasa si los datos son un conjunto de fotos y tenemos que determinar algo con base en la foto?
# ¿Cuáles son las observaciones y las características en ese caso?

# Supongamos que la observación es una foto y cada pixel es una característica de esa observación
# 1. la imagen se convierte en un vector de características:
#     cada pixel se convierte en un número que representa su intensidad (en escala de grises) o su color (en RGB)
# 2. se entrena un modelo de aprendizaje automático con esos vectores de características:
#     el modelo aprende a reconocer patrones en los datos de entrenamiento, el conjunto de valores 
# 3. se utiliza el modelo para hacer predicciones sobre nuevas imágenes:

# Vamos a convertir los valores de píxel en un vector para obtener nuestras características
# Ejemplo: [255, 255, 255, 255, 237, 217, 239, 255, 255, 255, 255, 255, 255, 255, 255, 190, 75, 29, 29, 30, 81, 198, 255, 255, 255, 255, 255, 147, 30, 29, 29, 29, 29, 29, 31, 160, 255, 255, 255, 185, 29, 29, 29, 29, 29, 29, 29, 29, 31, 198, 255, 255, 61, 29, 29, 29, 29, 29, 29, 29, 29, 29, 74, 255, 255, 108, 121, 121, 121, 121, 121, 121, 121, 121, 121, 102, 219, 255, 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 238, 107, 168, 255, 238, 153, 150, 152, 244, 201, 152, 150, 178, 253, 103, 144, 248, 243, 121, 108, 114, 225, 184, 130, 112, 154, 235, 103, 62, 197, 255, 227, 168, 231, 149, 230, 196, 179, 251, 183, 29, 29, 105, 255, 255, 219, 195, 191, 184, 195, 235, 255, 91, 29, 29, 30, 187, 255, 234, 218, 218, 218, 218, 243, 174, 29, 29, 29, 29, 38, 180, 255, 255, 255, 255, 255, 169, 35, 29, 29, 29, 29, 29, 29, 82, 153, 174, 150, 76, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]

# el tamaño de cada imagen indicara el número de características que tendrá cada observación
# Ejemplo: 
#     una imagen de 28x28 píxeles tendrá 784 características (28x28=784)
#     una imagen de 1920x1080 pixeles tendrá 2,073,600 características (1920x1080=2,073,600)

# Los algoritmos clásicos como la potenciación del gradiente no pueden administrar el entrenamiento con tantas funciones.
# ya que la cantidad de características es tan grande, el modelo se vuelve muy complejo y difícil de entrenar
# además, la mayoría de las características (píxeles) no son relevantes para la tarea de clasificación, lo que puede llevar a un sobreajuste del modelo a los datos de entrenamiento

# qué tienen en común las imágenes y los textos:
#     1. Ambos tienen información redundante: ciertos pixeles son menos importantes que otros, esto va similar a las palabras que no tienen tanto significado en un texto
#     2. Las características vecinas están relacionadas entre sí: A menudo, los pixeles adyacentes pertenecen al mismo objeto y las palabras vecinas en una oración tienen un significado relacionado
#     3. Mezclar columnas de datos tabulares no cambia su significado, mientras que mezclar pixeles en una imagen producirá ruido.
#     4. Mezclar palabras en un texto también producirá ruido, ya que el orden de las palabras es importante para el significado de la oración
#     5. Las imágenes y los textos tienen una estructura jerárquica: en las imágenes, los píxeles forman bordes, los bordes forman formas y las formas forman objetos; en los textos, las letras forman palabras, las palabras forman oraciones y las oraciones forman párrafos

# Las redes neuronales ayudan a procesar una gran cantidad de características.
# Las redes neuronales pueden tener en cuenta el orden de las características.

# Las redes neuronales muestran mejores resultados para:
#     1. Vídeo(conjunto de imagenes)
#     2. Pistas de audio.
#     3. Robots de chat.
# en comparación con la regresión lineal o la potenciación del gradiente

#---------------------------- tipos de problemas de machine learning ----------------------------
# Clasificación:
# - Predecir categorías discretas
# - Ejemplo: "niño", "adolescente", "adulto", "anciano"

# Regresión:
# - Predecir valores numéricos continuos
# - Ejemplo: edad exacta como 25, 34, 67 años

#---------------------------- tipos de modelos de machine learning ----------------------------
# Modelos de Regresión
# Para predecir valores numéricos continuos:
#     Regresión Lineal: Relación lineal entre variables
#     Regresión Polinomial: Relaciones no lineales
#     Regresión Logística: Para clasificación binaria
#     Ridge/Lasso: Con regularización

# Modelos basados en Árboles
# Estructuras de decisión:
#     Árbol de Decisión: Un solo árbol
#     Random Forest: Múltiples árboles (ensemble)
#     Gradient Boosting: XGBoost, LightGBM, CatBoost
#     Extra Trees: Árboles extremadamente aleatorios

# Modelos de Ensemble (Conjunto)
# Combinan múltiples modelos:
#     Bagging: Random Forest, Extra Trees
#     Boosting: AdaBoost, Gradient Boosting
#     Stacking: Combina diferentes tipos de modelos
#     Voting: Voto mayoritario de varios modelos

# Modelos de Redes Neuronales
# Inspirados en el cerebro:
#     Perceptrón: Modelo más simple
#     Redes Neuronales Profundas (Deep Learning)
#     CNN: Para imágenes
#     RNN/LSTM: Para secuencias y texto
#     Transformers: GPT, BERT

# Modelos basados en Distancia
# Usan proximidad entre datos:
#     K-Nearest Neighbors (KNN): Vecinos más cercanos
#     K-Means: Para clustering
#     SVM: Support Vector Machines

# Modelos Probabilísticos
# Basados en probabilidades:
#     Naive Bayes: Para clasificación de texto
#     Modelos de Mezcla Gaussiana
#     Análisis Discriminante

#---------------------------- Libreria keras ----------------------------
# una librería de red neuronal de código abierto.
# regresión lineal en Keras(una regresión lineal es una red neuronal con una sola neurona):

# Importacin de los datos, hasta aqui todo es igual
import pandas as pd
data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop('target', axis=1)
target_train = data_train['target']

data_valid = pd.read_csv('/datasets/test_data_n.csv')
features_valid = data_valid.drop('target', axis=1)
target_valid = data_valid['target']

# Creacion desde sklearn, modo aprendido antes
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(features_train, target_train)

# Creacion desde Keras
from tensorflow import keras
model = keras.models.Sequential()
    # Inicializa el modelo (la red neuronal)
    # Establezcamos la clase del modelo en Sequential (se utilizada para modelos con capas secuenciales)
    # Una capa es un conjunto de neuronas que comparten la misma entrada y salida.
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1])) # indica cómo está organizada la red neuronal
    # crea una capa de neuronas
    # "Dense" significa que cada entrada estará conectada a cada neurona o salida
    # units establece la cantidad de neuronas en la capa
    # input_dim establece la cantidad de entradas en la capa
    # aqui no seconsidera el sesgo
model.compile(loss='mean_squared_error', optimizer='sgd') # indica cómo está entrenada la red neuronal
    # Especifica el ECM como la función de pérdida de la tarea de regresión para el parámetro loss
    # Establece el método de descenso de gradiente para el parámetro Optimizer
model.fit(features_train, target_train, verbose=2, epochs=5, validation_data=(features_valid, target_valid) )
# verbose=2 muestra una barra de progreso para cada época, pero sin mostrar la pérdida de cada época
# epochs = 5 indica que el modelo pasará por todo el conjunto de datos 5 veces durante el entrenamiento
#     El valor del ECM disminuye con cada época.
#     Cuando entrenamos una red neuronal, generalmente ejecutamos muchas épocas. 
#     Detenemos el entrenamiento cuando se alcanza la calidad deseada del modelo o cuando el ECM deja de disminuir.
#     validation_data=(features_valid, target_valid) se utiliza para evaluar el rendimiento del modelo en un conjunto de datos de validación después de cada época.

# Tiene n entradas, cada una multiplicada por su propio peso. 
# Su peso se designa como b (sesgo).
# Lo que constituye el proceso de entrenamiento de la red neuronal es la selección de pesos w y b
# Después de sumar todos los productos de los valores de las entradas y los pesos, la respuesta de la red neuronal (a) se envía a la salida.
# Cada conexión tiene su propio peso (w)
# Cada neurona tiene su propio sesgo (b)
# La función de activación introduce no-linealidad
# Múltiples capas permiten patrones complejos
# Las capas en las que todas las entradas están conectadas a todas las neuronas se denominan capas totalmente conectadas.

#---------------------------- redes neuronales ----------------------------
# Las redes neuronales son muy flexibles en cuanto a entradas y salidas
# No siempre proporcionales las entradas y salidas

# Ejemplos donde NO coinciden:
#     Entrada: 1000 píxeles de imagen → Salida: 1 etiqueta ("gato" o "perro")
#     Entrada: 500 palabras de texto → Salida: 1 sentimiento ("positivo" o "negativo")

# ¿Cómo funciona internamente?
# Entrada (X datos) → Capas Ocultas → Salida (Y datos)

# Las capas ocultas pueden:
# Reducir dimensiones: 1000 → 100 → 10 → 1
# Aumentar dimensiones: 10 → 100 → 1000
# Mantener dimensiones: 100 → 100 → 100

# Las redes neuronales adaptan el número de salidas según el problema que quieres resolver, no según el número de entradas.

#---------------------------- Diagrama de Red Neuronal Simple ----------------------------
# Entradas    Pesos    Suma Ponderada    Sesgo    Función Activación    Salida
#    x₁  ────→  w₁  ────→     \              +b  ────→    f(z)    ────→   y
#    x₂  ────→  w₂  ────→      Σ(xi*wi)     /
#    x₃  ────→  w₃  ────→     /

# Paso 1: Multiplicación
# x₁ * w₁ = resultado₁
# x₂ * w₂ = resultado₂  
# x₃ * w₃ = resultado₃

# Paso 2: Suma
# z = (x₁*w₁) + (x₂*w₂) + (x₃*w₃)

# Paso 3: Agregar sesgo
# z = z + b

# Paso 4: Función de activación
# y = f(z)  [ejemplo: sigmoid, ReLU, tanh]

# Entrada: x₁=2, x₂=3, x₃=1
# Pesos: w₁=0.5, w₂=0.3, w₃=0.8
# Sesgo: b=0.1
# Paso 1: z = (2×0.5) + (3×0.3) + (1×0.8) + 0.1
#         z = 1.0 + 0.9 + 0.8 + 0.1 = 2.8
# Paso 2: y = sigmoid(2.8) = 0.94

#---------------------------- Diagrama de Capas Totalmente Conectadas ----------------------------
# ENTRADA (Input Layer)
#     ↓
# [x₁] [x₂] [x₃] ... [xₙ]
#  │    │    │       │
#  │    │    │       │
#  └────┼────┼───────┘
#       │    │    
#       │    │    ┌─────────────────┐
#       │    └────│  CAPA DENSA     │
#       │         │  (Dense Layer)  │
#       └─────────│                 │
#                 │  • Cada neurona │
#                 │    conectada a  │
#                 │    TODAS las    │
#                 │    entradas     │
#                 └─────────────────┘
#                         ↓
#               [n₁] [n₂] [n₃] ... [nₘ]
#                │    │    │       │
#                │    │    │       │
#                └────┼────┼───────┘
#                     │    │
#                     │    │  ┌─────────────────┐
#                     │    └──│  CAPA SALIDA    │
#                     │       │  (Output Layer) │
#                     └───────│                 │
#                             │  • 1 neurona    │
#                             │    (regresión)  │
#                             └─────────────────┘
#                                     ↓
#                                   [y]

# Cada neurona de una capa se conecta con TODAS las neuronas de la capa anterior
# No hay conexiones saltadas o parciales
# las neuronas en cada capa están conectadas con las de las capas adyacentes.
# todas las capas, excepto las capas de entrada y salida, se llaman capas ocultas

# Para obtener z, suma todos los productos de los valores de entrada y los pesos:
# z1 = x1 * w11 + x2 * w12
# z2 = x1 * w21 + x2 * w22
# z3 = z1 * w31 + z2 * w32

# Los sigmoides permiten hacer que la red sea más compleja.

#---------------------------- SGD/ DGE (Stochastic Gradient Descent/Descenso de Gradiente Estocástico) ----------------------------
# ¿Qué es?
#     - Es un optimizador
#     - Algoritmo que ajusta los pesos de la red neuronal
#     - Busca minimizar la función de p érdida (MSE)
# ¿Cómo funciona?
#     1. Calcula el error (MSE)
#     2. Encuentra la dirección para reducir el error
#     3. Ajusta los pesos un poquito en esa dirección
#     4. Repite hasta converger
#---------------------------- ¿Qué es una época? ----------------------------
# Una época es una pasada completa de todos los datos de entrenamiento a través de la red neuronal.

# 1 época = Leer todo una vez completa
# 5 épocas = Leer todo 5 veces seguidas
# Cada vez que lees, aprendes un poco más

# Época 1: La red ve todos los datos → ajusta pesos
# Época 2: La red ve todos los datos otra vez → ajusta más
# Época 3: La red ve todos los datos otra vez → sigue mejorando
# ...

# En cada época:
# Entrena con features_train y target_train
# Evalúa con features_valid y target_valid
# Muestra ambos resultados

# Te permite detectar sobreajuste (overfitting):
# Si loss baja pero val_loss sube → ¡Problema!
# Si ambos bajan juntos → ¡Bien!

#---------------------------- Regresión logística, Ques es diferente de la regresión lineal----------------------------
# La regresión lineal es una red neuronal con una sola neurona. Se puede decir lo mismo sobre la regresión logística.
# tiene una función sigmoide, o función logística, como función de activación
# toma cualquier número real como entrada y devuelve un número en el rango de 0 (sin activación) a 1 (activación).

# Este número en el rango de 0 a 1 se puede interpretar como una predicción de una red neuronal sobre si la observación pertenece a la clase positiva o a la clase negativa.
# La función de pérdida varía dependiendo del tipo de red neuronal.

# El ECM se usa en tareas de regresión
# la entropía cruzada binaria (BCE) es la adecuada para una clasificación binaria

# ENTRADA → PROCESAMIENTO → FUNCIÓN SIGMOIDE → SALIDA
#    ↓            ↓              ↓            ↓
#   x₁ ────┐                                 
#          │                                 
#   x₂ ────┼──→ [w₁x₁ + w₂x₂ + b] ──→ σ(z) ──→ Probabilidad
#          │         ↑                ↑         (0 a 1)
#   x₃ ────┘         z            Sigmoide

# 1. Entradas (x₁, x₂, x₃...):
# - Características de los datos
# 2. Combinación lineal (z):
# z = w₁x₁ + w₂x₂ + w₃x₃ + ... + b
# 3. Función Sigmoide σ(z):
# σ(z) = 1 / (1 + e^(-z))
# 4. Salida:
# - Probabilidad entre 0 y 1
# - Si > 0.5 → Clase 1
# - Si < 0.5 → Clase 0

# Regresión Lineal:
# x → [Combinación lineal] → Valor numérico

# Regresión Logística:
# x → [Combinación lineal] → [Sigmoide] → Probabilidad (0-1)

#---------------------------- Regresión logística ----------------------------
import pandas as pd
data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop('target', axis=1)
target_train = data_train['target']

data_valid = pd.read_csv('/datasets/test_data_n.csv')
features_valid = data_valid.drop('target', axis=1)
target_valid = data_valid['target']

from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=1, input_dim=features_train.shape[1], activation='sigmoid')) 
    # Aplica la función de activación a la capa totalmente conectada
model.compile(loss='binary_crossentropy', optimizer='sgd')
    # Especifica la entropía cruzada binaria como la función de pérdida para la tarea de clasificación binaria
model.fit(features_train, target_train, verbose=2, epochs=5, validation_data=(features_valid, target_valid) )

# ¿Qué es binary_crossentropy?
# Es una función que mide qué tan lejos están las predicciones de la realidad en problemas de clasificación con 2 clases (0 o 1).
# ¿Cómo funciona?
# Si la clase real es 1: pérdida = -log(probabilidad_predicha)
# Si la clase real es 0: pérdida = -log(1 - probabilidad_predicha)
# Ejemplo práctico:
# Caso 1: Predicción = 0.9, Real = 1 → Pérdida baja (¡bien!)
# Caso 2: Predicción = 0.1, Real = 1 → Pérdida alta (¡mal!)
# Caso 3: Predicción = 0.2, Real = 0 → Pérdida baja (¡bien!)

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score
df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']
df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=1, input_dim=features_train.shape[1], activation='sigmoid'
    )
)
model.compile(loss='binary_crossentropy', optimizer='sgd')
model.fit(
    features_train,
    target_train,
    epochs=5,
    verbose=0,
    validation_data=(features_valid, target_valid),
)
predictions = model.predict(features_valid) > 0.5
score = accuracy_score(predictions, target_valid)
print("Exactitud:", score)

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score
df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']
df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(
        units=1, input_dim=features_train.shape[1], activation='sigmoid'
    )
)
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['acc'])
model.fit(features_train, target_train, epochs=10, verbose=2,
          validation_data=(features_valid, target_valid))

#---------------------------- Cómo se entrenan las redes neuronales ----------------------------
# las redes multicapa están entrenadas con un descenso de gradiente
# Hay parámetros que son los pesos de las neuronas en cada capa totalmente conectada.
# Y el objetivo de entrenamiento es encontrar los parámetros que resultan en la función de pérdida mínima.

# puntuaciones de aprendizaje = pasos de descenso de gradiente
# usar un Learning rate bajo puede hacer que el entrenamiento sea muy lento, mientras que un Learning rate alto puede hacer que el entrenamiento diverja o no converja al mínimo de la función de pérdida.

# Cada neurona crea una linea recta que se puede combinar con otras para formar figuras
# Cada neurona con función sigmoide puede crear una frontera suave

# https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.003&regularizationRate=0&noise=0&networkShape=5&seed=0.79032&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

# esta es una tarea de clasificacion:
# Puntos azules y amarillos = Las dos CLASES que queremos separar
#     Puntos azules: Clase 0 (por ejemplo: "No compra el producto")
#     Puntos amarillos: Clase 1 (por ejemplo: "Sí compra el producto")

# ¿Qué es una "frontera suave"?
# La línea que separa las dos clases NO es una línea dura, sino gradual
#     Línea dura:    azul|amarillo  (cambio abrupto)
#     Frontera suave: azul~~~amarillo (cambio gradual)

# Los COLORES del fondo significan:
#     Área azul: "Aquí la red predice clase azul"
#     Área amarilla: "Aquí la red predice clase amarilla"
#     Gradiente de colores: Muestra la confianza de la predicción

# ¿Qué es la activación SIGMOIDE?
#     Convierte cualquier número en un valor entre 0 y 1
#     Crea curvas suaves (no líneas rectas)
#     Permite no-linealidad

# ¿Qué es la activación LINEAL?
#     No transforma el valor de entrada
#     Solo puede crear líneas rectas
#     Es completamente lineal

# El problema clave: Sin importar cuántas capas agregues con activación lineal, siempre obtienes una función lineal.

# Con activación lineal:
# 1 capa = línea recta
# 10 capas = ¡sigue siendo línea recta!
# 100 capas = ¡sigue siendo línea recta!

# Con activación sigmoide:
# 1 capa = línea curva
# 3 capas = formas complejas (como círculos)

# función de activación lineal = sin activación
# cualquier red totalmente conectada es esencialmente una función lineal. 
# Los datos no son lineales (es decir, las clases no se pueden separar en una línea recta).

# A medida que aumenta el número de capas, el entrenamiento se vuelve menos eficiente
# Cuantas más capas haya en la red, menos señal habrá de la entrada a la salida de la red. 
# Esto se llama señal de fuga y es causada por el sigmoide, que convierte grandes valores en más pequeños una y otra vez.

# Para deshacerte del problema puedes intentar otra función de activación, por ejemplo, ReLU
# ReLU lleva todos los valores negativos a 0 y deja valores positivos sin cambios.
# La forma depende de cómo se inicializaron los pesos de la red.
# La activación de ReLU permite a la red entrenar incluso con 6 capas. 
# No habrá ningún problema a menos que agregues un par de docenas de capas. 
# Esas serán necesarias para el procesamiento de imágenes

# Sigmoid tiene el problema de "señal de fuga" cuando usas muchas capas - la señal se debilita capa tras capa
# ReLU permite entrenar redes más profundas sin este problema

#---------------------------- ¿Qué es la función de pérdida mínima? ----------------------------
# Es el punto más bajo que puede alcanzar la función de pérdida durante el entrenamiento. 
# el menor error posible que tu modelo puede lograr.

# Imagina que estás en una montaña con los ojos vendados:
# Función de pérdida = Tu altura en la montaña
# Función de pérdida mínima = El valle más profundo
# Descenso de gradiente = Caminar hacia abajo paso a paso

# En el entrenamiento:
# Época 1: Pérdida = 0.8234
# Época 2: Pérdida = 0.7123  ← Bajando
# Época 3: Pérdida = 0.6891  ← Sigue bajando
# Época 4: Pérdida = 0.6890  ← Ya casi no baja
# Época 5: Pérdida = 0.6890  ← Mínimo alcanzado

#---------------------------- ¿Cómo cambia una red neuronal cuando agregamos neuronas y capas? ----------------------------
# Al agregar MÁS NEURONAS(Capacidad de aprendizaje):
#     Pocas neuronas: Modelo simple, puede no capturar patrones complejos
#     Más neuronas: Modelo más complejo, puede aprender patrones más sofisticados

# Al agregar MÁS CAPAS:
# Ventajas:
#     Mayor complejidad: Puede resolver problemas más difíciles
#     Patrones jerárquicos: Cada capa aprende características diferentes

# Desafíos:
# Señal de fuga: Con muchas capas, la información se "pierde"
# Entrenamiento más difícil: Especialmente con función sigmoide

# Problema de la función Sigmoide:
# Entrada → Capa 1 (sigmoide) → Capa 2 (sigmoide) → ... → Salida
#   100   →      0.9          →      0.7          → ... →  0.1
# Los valores se hacen cada vez más pequeños.

#---------------------------- redes neuronales totalmente conectadas con keras ----------------------------
# ¿Qué debemos hacer para construir una red multicapa totalmente conectada?
# Agregar una capa totalmente conectada varias veces

import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score
df = pd.read_csv('/datasets/train_data_n.csv')
df['target'] = (df['target'] > df['target'].median()).astype(int)
features_train = df.drop('target', axis=1)
target_train = df['target']
df_val = pd.read_csv('/datasets/test_data_n.csv')
df_val['target'] = (df_val['target'] > df['target'].median()).astype(int)
features_valid = df_val.drop('target', axis=1)
target_valid = df_val['target']
model = keras.models.Sequential()
model.add( #primera capa oculta, 10 neuronas, función de activación sigmoide
    keras.layers.Dense(
        units=10, input_dim=features_train.shape[1], activation='sigmoid'
    )
)
model.add( #segunda capa oculta, 1 neurona
    keras.layers.Dense(
        units=1, activation='sigmoid'
    )
)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
model.fit(features_train, target_train, epochs=10, verbose=2,
          validation_data=(features_valid, target_valid))

#resultado:
# Epoch 1/10
# 32/32 - 1s - loss: 0.7083 - acc: 0.5220 - val_loss: 0.7082 - val_acc: 0.5160 - 892ms/epoch - 28ms/step
# Epoch 2/10
# 32/32 - 0s - loss: 0.7065 - acc: 0.5250 - val_loss: 0.7084 - val_acc: 0.5120 - 123ms/epoch - 4ms/step
# Epoch 3/10
# 32/32 - 0s - loss: 0.7050 - acc: 0.5280 - val_loss: 0.7071 - val_acc: 0.5140 - 122ms/epoch - 4ms/step
# Epoch 4/10
# 32/32 - 0s - loss: 0.7034 - acc: 0.5290 - val_loss: 0.7071 - val_acc: 0.5110 - 98ms/epoch - 3ms/step
# Epoch 5/10
# 32/32 - 0s - loss: 0.7019 - acc: 0.5370 - val_loss: 0.7079 - val_acc: 0.5050 - 122ms/epoch - 4ms/step
# Epoch 6/10
# 32/32 - 0s - loss: 0.7006 - acc: 0.5330 - val_loss: 0.7084 - val_acc: 0.5060 - 101ms/epoch - 3ms/step
# Epoch 7/10
# 32/32 - 0s - loss: 0.6991 - acc: 0.5390 - val_loss: 0.7083 - val_acc: 0.5070 - 104ms/epoch - 3ms/step
# Epoch 8/10
# 32/32 - 0s - loss: 0.6977 - acc: 0.5420 - val_loss: 0.7076 - val_acc: 0.5060 - 96ms/epoch - 3ms/step
# Epoch 9/10
# 32/32 - 0s - loss: 0.6965 - acc: 0.5430 - val_loss: 0.7063 - val_acc: 0.5090 - 124ms/epoch - 4ms/step
# Epoch 10/10
# 32/32 - 0s - loss: 0.6953 - acc: 0.5500 - val_loss: 0.7054 - val_acc: 0.5130 - 100ms/epoch - 3ms/step

#---------------------------- trabajar con imagenes en python ----------------------------
# Ya sabes que las imágenes son conjuntos de números. 
# Si la imagen es en blanco y negro, cada pixel almacena un número de 0 (negro) a 255 (blanco).

#importacion de liberias
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#abrir imagen
image = Image.open('/datasets/ds_cv_images/face.png')

#convertir a matriz NumPy
array = np.array(image)
print(array) 

#mostrar imagen 
plt.imshow(array)
    #el mapa de color predeterminado no es blanco y negro

#repintar celdas segun cordenadas en el array
array[0, 0] = 0 #esquina superior izquierda, negro
array[14, 12] = 255 #esquina inferior derecha, negro

# las redes neuronales aprenden mejor cuando reciben imágenes en el rango de 0 a 1 como entrada
array = array / 255.

#establecer el mapa de color en blanco y negro
plt.imshow(array, cmap='gray')
plt.colorbar() #señalar los valores numericos en el rango de color blanco y negro

#---------------------------- imagenes en color ----------------------------
# Las imágenes en color o imágenes RGB consisten en tres canales: rojo, verde y azul.
# dichas imágenes son matrices tridimensionales con celdas que contienen númerosango de 0 a 255.
fila, columna, canal

np.array([[0, 255], #bidimencional
          [255, 0]])

np.array([[[0, 255, 0], [128, 0, 255]], #tridimencional
          [[12, 89, 0], [5,  89, 245]]])

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image = Image.open('/datasets/ds_cv_images/cat.jpg')
array = np.array(image)
print(array.shape) # (altura, anchura, canales)
    # Resultado: (300, 400, 3)
red_channel = array[:,:,0] #solo canal rojo
plt.imshow(red_channel)
plt.colorbar()

#---------------------------- Clasificación multiclase ----------------------------
# implica que las observnaciones pertenecen a una de varias clases en vez de a una de dos clases.
# Cada neurona es responsable de su propia clase
# la capa de salida puede tener mas de 1 neurona
# la cantidad de clases no afectará el cálculo de la función de pérdida

# p es la probabilidad de respuesta correcta devuelta por la red.

# La función de activación que se adapta a este caso se llama SoftMax,:
#     toma varias salidas de las redes y devuelve probabilidades que son todas iguales a uno.

# Así es como calculamos las probabilidades:
# p1 = SoftMax(z1) = e^z1 / (e^z1 + e^z2 + e^z3)
# p2 = SoftMax(z2) = e^z2 / (e^z1 + e^z2 + e^z3)
# p3 = SoftMax(z3) = e^z3 / (e^z1 + e^z2 + e^z3)
# Ahora p₁ varía en el rango de 0 a 1.

# si z₁ es significativamente mayor que z₂ y z₃, entonces el numerador es aproximadamente igual al denominador
# Si z₁ es significativamente menor que z₂ o z₃, entonces el numerador es mucho menor que el denominador

# Ahora las probabilidades suman uno:
# p1 + p2 + p3 = SoftMax(z1) + SoftMax(z2) + SoftMax(z3) =
# = e^z1 / (e^z1 + e^z2 + e^z3) + e^z2 / (e^z1 + e^z2 + e^z3) + e^z3 / (e^z1 + e^z2 + e^z3) =
# = (e^z1 + e^z2 + e^z3) / (e^z1 + e^z2 + e^z3) = 1

# ¿Por qué el bloque SoftMax del diagrama depende de todas las salidas de la red? 
# Porque realmente necesitamos todos los resultados para encontrar todas las probabilidades
# El número de neuronas en la capa de salida debe ser igual al número de clases y todas las salidas deben estar conectadas a SoftMax
# Las probabilidades de SoftMax en la etapa de entrenamiento pasarán a la entropía cruzada, la cual calculará el error
# La función de pérdida se minimizará utilizando el método de descenso de gradiente
# La única condición para que funcione es que la función tenga una derivada para todos los parámetros: pesos y sesgo de la red neuronal.

# Elementos clave del diagrama:
# Entrada:
# - x₁, x₂ → características de entrada

# Capas ocultas:
# - Neuronas intermedias con sus pesos y conexiones

# Capa de salida:
# - z₁, z₂, z₃ → salidas brutas (logits) para cada clase

# SoftMax:
# - Bloque SoftMax que toma TODAS las salidas (z₁, z₂, z₃)
# - p₁, p₂, p₃ → probabilidades finales que suman 1

# Esta es la inicialización de la última capa para la clasificación binaria:
Dense(units=1, activation='sigmoid')
#     Esta línea está creando la capa de salida de una red neuronal para clasificación binaria.
#     En clasificación binaria solo necesitas distinguir entre 2 clases

# La función sigmoid hace:
#     Convierte cualquier número en un valor entre 0 y 1
#     Este valor se puede interpretar como una probabilidad
#     Por ejemplo: 0.8 significa 80% de probabilidad de ser la Clase A

# Y esta es la inicialización para la clasificación multiclase:
Dense(units=3, activatio='softmax')

#---------------------------- ¿Qué es z1? ----------------------------
# z1 es el valor crudo (sin procesar) que sale de la neurona 1 en la capa de salida
# antes de aplicar cualquier función de activación. 

# Qué significa "número positivo grande"?
# Imagina que tienes 3 clases y obtienes estos valores:
# z1 = 8.5   ← ¡Número positivo GRANDE!
# z2 = 1.2
# z3 = 0.3

# ¿Cómo decide la red la clase?
# Paso 1: La función SoftMax convierte estos valores en probabilidades:
# p1 = e^8.5 / (e^8.5 + e^1.2 + e^0.3) ≈ 0.99  ← ¡Casi 100%!
# p2 = e^1.2 / (e^8.5 + e^1.2 + e^0.3) ≈ 0.007
# p3 = e^0.3 / (e^8.5 + e^1.2 + e^0.3) ≈ 0.003

# Paso 2: La red elige la clase con mayor probabilidad → Clase 1

# En términos simples:
# "Número positivo grande en z1" = "La red está MUY segura de que es Clase 1"

# Ejemplo:
# Si clasificas imágenes de animales:
# z1 = 9.2  (perro)   ← ¡MUY seguro!
# z2 = 0.1  (gato)
# z3 = -1.5 (pájaro)
# La red dice: "¡Estoy 99.8% seguro de que es un PERRO!"

# ¿Y si z1 fuera negativo grande?
# z1 = -8.5  ← Número negativo grande
# z2 = 2.0
# z3 = 1.0
# Entonces p1 ≈ 0.001 → "Definitivamente NO es Clase 1"

# esto se repite para todas las clases en la clasificación multiclase. 

# En el ejemplo que estás viendo:
# z1 representa la salida para la clase 1
# z2 representa la salida para la clase 2
# z3 representa la salida para la clase 3
# Y así sucesivamente para tantas clases como tengas en tu problema.

# cada neurona en la capa de salida es responsable de una clase específica

#---------------------------- ¿Qué es la entropía cruzada? ----------------------------
# es una función de pérdida que mide qué tan "lejos" están las predicciones de tu red de las respuestas correctas

# Fórmula básica:
# CE = -log(p)
# Donde p es la probabilidad de la respuesta correcta.

# Caso 1 - Predicción muy buena:

# p = 0.95  (95% seguro que es gato)
# CE = -log(0.95) = 0.05  ← Error PEQUEÑO
# Caso 2 - Predicción muy mala:

# p = 0.01  (1% seguro que es gato)  
# CE = -log(0.01) = 4.6   ← Error GRANDE

# ¿Por qué usar entropía cruzada?
#     Penaliza más las predicciones muy incorrectas
#     Tiene derivadas (necesario para el descenso de gradiente)
#     Funciona bien con probabilidades (SoftMax)

#---------------------------- clasificacion de imagenes ----------------------------
# clasificador de diez clases para el agregador de tiendas de ropa
# Cada clase representará una categoría de producto

# Etiqueta	Descripción
# 0	         Camiseta/top
# 1	         Pantalón
# 2	         Suéter
# 3	         Vestido
# 4	         Abrigo
# 5	         Sandalia
# 6	         Camisa
# 7	         Zapato deportivo
# 8	         Bolsa
# 9	         Botín
# El modelo debe tomar fotos del producto en blanco y negro como entrada

from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
(features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
print('Train:', features_train.shape)
print('Test:', features_test.shape)
print('First image class:', target_train[0])
plt.imshow(features_train[0], cmap='gray')

# En una red totalmente conectada, las observaciones de entrada deben ser las filas de la tabla y todo el conjunto de datos debe ser una tabla bidimensional.

from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
(features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
features_train = features_train.reshape(features_train.shape[0], 28 * 28)
features_test = features_test.reshape(features_test.shape[0], 28 * 28)
print("Train:", features_train.shape)
print("Test:", features_test.shape)

# El problema: Imágenes vs Redes Neuronales
#     Las imágenes tienen forma 3D (alto × ancho × canales)
#     las redes neuronales totalmente conectadas necesitan datos en formato 2D (filas × columnas).

# Antes del reshape:
#     features_train.shape = (60000, 28, 28)
# Esto significa:
# - 60,000 imágenes
# - Cada imagen es de 28 × 28 píxeles

# Después del reshape:
#     features_train.shape = (60000, 784)
# Esto significa:
# - 60,000 filas (una por imagen)
# - 784 columnas (28 × 28 = 784 píxeles por imagen)

# ¿Cómo funciona el reshape?
#     features_train.reshape(features_train.shape[0], 28 * 28)
#     features_train.shape[0] → mantiene el número de imágenes (60,000)
#     28 * 28 → convierte cada imagen 28×28 en una fila de 784 píxeles

from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
(features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
features_train = features_train.reshape(features_train.shape[0], 28 * 28)
features_test = features_test.reshape(features_test.shape[0], 28 * 28)
model = keras.models.Sequential()
model.add(
    keras.layers.Dense(units=10, input_dim=28 * 28, activation='softmax')
)
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc']
)
model.fit(
    features_train,
    target_train,
    epochs=1,
    verbose=2,
    validation_data=(features_test, target_test),
)

# El entrenamiento de regresión logística es un proceso lento
# loss='sparse_categorical_crossentropy: es la función de pérdida adecuada para clasificación multiclase con etiquetas enteras
#     Es entropía cruzada para clasificación multiclase 
#     Significa que tus etiquetas están en formato simple (números), no en formato one-hot

# Formato sparse (lo que tienes):
# target = [0, 1, 2, 9, 5]  # Números simples

# Formato one-hot (lo que NO tienes):
# target = [[1,0,0,0,0,0,0,0,0,0],  # Clase 0
#           [0,1,0,0,0,0,0,0,0,0],  # Clase 1
#           [0,0,1,0,0,0,0,0,0,0]]  # Clase 2

# input_dim=28 * 28
#     le dice a la primera capa cuántas características de entrada esperar
#     28 * 28 = 784 → cada imagen tiene 784 píxeles
#     Es como decirle a la red: "Espera recibir 784 números por cada imagen"

#---------------------------- Entrenamiento de red multicapa ----------------------------
# Las redes neuronales multicapa sirven como poderosas herramientas capaces de discernir patrones y características en las imágenes.
# Ofrecen una solución escalable y eficiente a la compleja tarea de clasificación de imágenes

# Paso 1. Generar es una función capaz de cargar tus datos en tu entorno 
#     Tomar una ruta donde se almacenan los datos.
#     Almacenar imágenes y etiquetas en diferentes arrays.
#     Realizar un preprocesamiento de las imágenes.
#     Dividir el conjunto de datos en un conjunto de datos de entrenamiento y uno de prueba y devolverlo.

# Ejercicio 1: Cargar y visualizar el dataset MNIST
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def read_idx(filename): #Leer archivos IDX y convertirlos en arrays de NumPy.
    with open(filename, 'rb') as f: #Abrir el archivo en modo binario
        zero, data_type, dims = struct.unpack('>HBB', f.read(4)) #Leer los primeros 4 bytes para obtener el tipo de datos y la cantidad de dimensiones
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims)) #Leer el tamaño de cada dimensión, creando una tupla con la forma del array
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape) #lee todo el resto del archivo, Luego numpy lo convierte en un array
            #el resultado es: (num_imagenes, alto, ancho)
def load_data(path_imgs, path_labels): 
    X = read_idx(path_imgs)
    y = read_idx(path_labels)
    X= X.astype('float32') #volver a decimales
        # Las redes neuronales trabajan mejor con números decimales que con enteros.
    X /= 255 #normalizar los valores de píxeles al rango [0, 1]
        # Las redes neuronales aprenden mejor cuando los datos están en el rango 0-1 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, y_train, X_test, y_test
imgs = 'archivos/train-images.idx3-ubyte'
labels = 'archivos/train-labels.idx1-ubyte'
X_train, y_train, X_test, y_test = load_data(imgs, labels)
print('Dataset shape:', X_train.shape[0], "[n_images,hight,width]")
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Ejercicio 2: Generar una red neuronal multicapa
from tensorflow.keras.models import Sequential #tipo de modelo donde las capas se apilan una después de otra
from tensorflow.keras.layers import Dense, Flatten
    # Dense: Capa totalmente conectada, cada neurona se conecta con todas las neuronas de la capa anterior
    # Flatten: Convierte matrices en vectores.
from tensorflow.keras.optimizers import Adam
    # Ajusta los pesos de la red para minimizar el error, uno de los más usados en deep learning.
def create_model(input_shape, num_classes): #función para crear el modelo
    model = Sequential() #definir el modelo como secuencial
    model.add(Flatten(input_shape=input_shape)) #aplanar la imagen de entrada (28, 28) a un vector de 784 elementos
    model.add(Dense(64, activation='relu')) #numero de neuronas / funcion de activacion 
    # (relu es una función de activación que introduce no linealidad, permitiendo a la red aprender patrones complejos)
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Capa de salida
    # Número de neuronas = número de clases / Softmax convierte los valores en probabilidades, la más alta es la predicción
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  #prepara el modelo para entrenar:
                  #Usa el optimizador Adam con una tasa de aprendizaje de 0.001,
                  loss='sparse_categorical_crossentropy',
                  #La funcion de perdida mide qué tan equivocada está la red
                  metrics=['acc'])
                  #Define qué medir durante entrenamiento y evaluación (accuracy en este caso)
    return model
create_model((28, 28), 10).summary() #Forma de los datos de entrada / Número de clases a predecir / Muestra la arquitectura.

# Ejercicio 3: Entrenar un modelo de red neuronal utilizando datos de entrenamiento y evaluar con datos de prueba.
def train_model(model,train_data, test_data, batch_size=32, epochs=5):
    # batch_size: el tamaño del lote para el entrenamiento (de forma predeterminada es 32).
    history = model.fit(train_data[0],train_data[1], #independientes
                    #train_data es una tupla que contiene dos elementos: train_data = (X_train, y_train)
                    # Posición 0: Las características (imágenes)
                    # Posición 1: Las etiquetas (clases)
                    validation_data=test_data, #en tupla
                    batch_size=batch_size, 
                    epochs=epochs,
                    shuffle=True, #Mezlar los datos de entrenamiento
                    verbose=2)
    return history
imgs = 'archivos/train-images.idx3-ubyte'
labels = 'archivos/train-labels.idx1-ubyte'
X_train, y_train, X_test, y_test = load_data(imgs, labels)
model = create_model((28, 28), 10)
model_trained = train_model(model, 
                        (X_train,y_train),
                        (X_test,y_test),
                        batch_size=32,
                        epochs=10,
                        verbose=1)

#---------------------------- Pipeline simple en redes neuronales para imágenes ----------------------------
# Imagen (28x28)
#      ↓
# Flatten
#      ↓
# Dense
#      ↓
# Dense
#      ↓
# Dense
#      ↓
# Softmax

Flatten(input_shape=(28,28))
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(10, activation='softmax')

# Problema de este enfoque:
#     este método solo funciona bien en datasets simples como MNIST.
#     Se pierde la estructura espacial de la imagen.

#---------------------------- Pipeline moderno para imágenes (CNN) ----------------------------
# Imagen
#  ↓
# Normalización
#  ↓
# Convolution
#  ↓
# Activation
#  ↓
# Pooling
#  ↓
# Convolution
#  ↓
# Pooling
#  ↓
# Flatten
#  ↓
# Dense
#  ↓
# Softmax

Conv2D(32)
ReLU
MaxPooling
Conv2D(64)
ReLU
MaxPooling
Flatten
Dense(128)
Dense(10)

# Las convoluciones detectan patrones visuales.
#     Primero detectan: bordes, líneas, curvas
#     Luego combinan eso para detectar: partes de objetos
#     finalmente: el objeto completo
#     pixeles → bordes → formas → objetos 

#---------------------------- Pipeline con regularización ----------------------------
# Cuando el dataset es grande se añaden técnicas para evitar overfitting
# Dropout apaga neuronas aleatoriamente para que el modelo no memorice los datos

# Imagen
#  ↓
# Conv
#  ↓
# Conv
#  ↓
# Pooling
#  ↓
# Dropout
#  ↓
# Flatten
#  ↓
# Dense
#  ↓
# Dropout
#  ↓
# Dense

#---------------------------- Convolución ----------------------------
# Las redes neuronales completamente conectadas no pueden trabajar con imágenes grandes
# Si no hay suficientes neuronas, la red no podrá encontrar todas las conexiones, y si hay demasiadas, la red estará sobreajustada.

# La convolución es una solución simple para esto.
# la convolución aplica las mismas operaciones a todos los pixeles.
# Para encontrar los elementos más importantes para la clasificación

# La convolución reduce el número de neuronas porque la misma operación se hará en cada bloque de la imagen.
# la red trabajará más rápido y evitará entrenarse en exceso.

# La convolución es como usar una "lupa especial" que busca patrones específicos en toda la imagen, pero usando la misma lupa en cada parte.

# Imagina que quieres encontrar contornos horizontales (como la suela de un zapato):
# Patrón que buscas:
# [0, 0, 0]  ← Fondo oscuro
# [1, 1, 1]  ← Suela brillante  
# [1, 1, 1]  ← Suela brillante
# [0, 0, 0]  ← Fondo oscuro
# En lugar de crear miles de neuronas diferentes, usas una sola "lupa" que se mueve por toda la imagen buscando este patrón.

#---------------------------- convolución unidimensional ----------------------------
# s = secuencia (los datos)
# w = pesos (el "filtro" o "lupa")
# c = convolución (el resultado)

# La convolución (c) se hace así: 
#     los pesos (w) se mueven a lo largo de la secuencia (s) y se calcula el producto escalar para cada posición en la secuencia.
# El filtro w se "desliza" sobre la secuencia s y calcula el producto punto en cada posición.
# s0 s1 s2 s3 s4 s5 s6 s7 ... Sm-1
# w0 w1 w2 ... ... Wn-1
# C0 = w0*s0 + w1*s1 + w2*s2 ...
# C1 = w0*s1 + w1*s2 + w2*s3 ...
# C2 = w0*s2 + w1*s3 + w2*s4 ...

# La duración n del vector de los pesos nunca es mayor que la duración m del vector de la secuencia
# de otro modo, no habría una posición a la que se pudiera aplicar la convolución.

# t es el índice para calcular el producto escalar y k es un valor entre 0 y (m - n + 1)
# Se escoge el número (m - n + 1) para que los pesos no excedan a la secuencia.

import numpy as np
def convolve(sequence, weights):
    convolution = np.zeros(len(sequence) - len(weights) + 1)
    for i in range(convolution.shape[0]):
        convolution[i] = np.sum(
            np.array(weights) * np.array(sequence[i : i + len(weights)])
        )
    return convolution

s = [2, 3, 5, 7, 11]
w = [-1, 1]

print(convolve(s, w))
# Resultado: [1. 2. 4. 4.]

# Tienes:
s = [2, 3, 5, 7, 11]  # Secuencia
w = [-1, 1]           # Pesos (filtro)

# Paso 1: Posición 0
# s: [2, 3, 5, 7, 11]
# w: [-1, 1]
#     ↑  ↑
# Cálculo: (-1 × 2) + (1 × 3) = -2 + 3 = 1

# Paso 2: Posición 1
# s: [2, 3, 5, 7, 11]
# w:    [-1, 1]
#        ↑  ↑
# Cálculo: (-1 × 3) + (1 × 5) = -3 + 5 = 2

# Paso 3: Posición 2
# s: [2, 3, 5, 7, 11]
# w:       [-1, 1]
#           ↑  ↑
# Cálculo: (-1 × 5) + (1 × 7) = -5 + 7 = 2

# Paso 4: Posición 3
# s: [2, 3, 5, 7, 11]
# w:          [-1, 1]
#              ↑  ↑
# Cálculo: (-1 × 7) + (1 × 11) = -7 + 11 = 4

# Resultado final:
# c = [1, 2, 2, 4]

#---------------------------- convolución bidimensional (2D) ----------------------------
# imagen bidimensional s con un tamaño de m×m pixeles y una matriz de peso de n×n pixeles. 
# La matriz es el kernel (núcleo) de la convolución.

# El kernel se mueve en la imagen de izquierda a derecha y de arriba a abajo. 
# Sus pesos se multiplican por cada pixel en todas las posiciones. 
# Los productos se suman y se registran como los pixeles resultantes.

s = [[1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 1, 1],
     [0, 0, 1, 1, 0],
     [0, 1, 1, 0, 0]]
w = [[1, 0, 1],
     [0, 1, 0],
     [1, 0, 1]]

# Imagen: 5×5
# Kernel: 3×3
# Resultado: (5-3+1) × (5-3+1) = 3×3

# Calculemos paso a paso:
# Posición 1: Esquina superior izquierda

# Región de la imagen:
[1, 1, 1]
[0, 1, 1]
[0, 0, 1]

# Kernel:
[1, 0, 1]
[0, 1, 0]
[1, 0, 1]

# Cálculo:
# (1×1) + (1×0) + (1×1) +
# (0×0) + (1×1) + (1×0) +
# (0×1) + (0×0) + (1×1) = 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 = 4

# Posición 2: Moviéndose una columna a la derecha

# Región de la imagen:
[1, 1, 0]
[1, 1, 1]
[0, 1, 1]

# Cálculo:
# (1×1) + (1×0) + (0×1) +
# (1×0) + (1×1) + (1×0) +
# (0×1) + (1×0) + (1×1) = 1 + 0 + 0 + 0 + 1 + 0 + 0 + 0 + 1 = 3

# Posición 3: Esquina superior derecha

# Región de la imagen:
[1, 0, 0]
[1, 1, 0]
[1, 1, 1]

# Cálculo:
# (1×1) + (0×0) + (0×1) +
# (1×0) + (1×1) + (0×0) +
# (1×1) + (1×0) + (1×1) = 1 + 0 + 0 + 0 + 1 + 0 + 1 + 0 + 1 = 4

# Continuando con todas las posiciones:
# Si calculamos todas las 9 posiciones (3×3), el resultado sería:
Resultado = [[4, 3, 4],
             [2, 4, 3],
             [2, 3, 4]]

# Puedes hallar los horizontales usando una convolución con el siguiente kernel:
np.array([[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]])
# encontrar los contornos de una imagen

# Usa el siguiente kernel para encontrar contornos verticales:
np.array([[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]])

import numpy as np
def convolve(sequence, weights):
    sequence = np.array(sequence)
    weights = np.array(weights)
    # Handle 1D convolution
    if sequence.ndim == 1 and weights.ndim == 1:
        convolution = np.zeros(len(sequence) - len(weights) + 1)
        for i in range(convolution.shape[0]):
            convolution[i] = np.sum(
                weights * sequence[i : i + len(weights)]
            )
        return convolution
    # Handle 2D convolution
    elif sequence.ndim == 2 and weights.ndim == 2:
        seq_height, seq_width = sequence.shape
        w_height, w_width = weights.shape
        
        conv_height = seq_height - w_height + 1
        conv_width = seq_width - w_width + 1
        convolution = np.zeros((conv_height, conv_width))
        for i in range(conv_height):
            for j in range(conv_width):
                convolution[i, j] = np.sum(
                    weights * sequence[i : i + w_height, j : j + w_width]
                )
        return convolution
    else:
        raise ValueError("Input dimensions must be either both 1D or both 2D")
s = [[2, 1, 3], 
     [4, 0, 2], 
     [1, 5, 6]]
w = [[-1, -2], 
     [1,   2]]
print(convolve(s, w))
# Resultado = [[0, -3], [7, 13]]

# Tu resultado es una matriz 2×2, lo que significa que el kernel se aplicó en 4 posiciones diferentes sobre la imagen original

# ### Posición [0,0] = 0
# Esquina superior izquierda
# Resultado: 0
# Esto significa que cuando el kernel se aplicó en la esquina superior izquierda de la imagen
# el producto punto dio 0.

# ### Posición [0,1] = -3
# Esquina superior derecha  
# Resultado: -3
# El kernel detectó algo que produjo un valor negativo (-3).

# ### Posición [1,0] = 7
# Esquina inferior izquierda
# Resultado: 7
# Aquí el kernel encontró un patrón que coincide fuertemente (valor positivo alto).

# ### Posición [1,1] = 13
# Esquina inferior derecha
# Resultado: 13
# Esta es la coincidencia más fuerte 
# el patrón del kernel coincide muy bien con esta región de la imagen.

# - Valores positivos altos (7, 13): El kernel encontró el patrón que está buscando
# - Valor cero (0): No hay coincidencia del patrón
# - Valor negativo (-3): El patrón es "opuesto" al que busca el kernel

#---------------------------- Capas convolucionales ----------------------------
# Las capas convolucionales aplican una operación de convolución a imágenes de input y hacen la mayor parte del cómputo dentro de la red.

# Una capa convolucional consiste en filtros (conjuntos de pesos) que se pueden personalizar y entrenar, los cuales se aplican a la imagen.
# un filtro es una matriz cuadrada con tamaño de K×K pixeles.

# Si el input es una imagen de color, se agrega al filtro una tercera dimensión, la profundidad.
# el filtro ya no es una matriz, sino un tensor (matriz multidimensional).

# hay tres canales de colores: rojo, azul y verde. 
# Un filtro de 3x3x3 (tres pixeles de ancho, alto y profundidad) se mueve a través de la imagen input en cada canal
# realizando una operación de convolución
# No se mueve a través de la tercera dimensión y cada color tiene pesos diferentes.
# las imágenes resultantes se doblan hacia el resultado final de la convolución.

# Dimensiones de la imagen:
# Imagen = Ancho × Alto × Profundidad
# Ejemplo: 100 × 100 × 3
# Ancho: 100 píxeles
# Alto: 100 píxeles
# Profundidad: 3 canales (RGB)

# Ejemplo:
# Canal Rojo: Filtro_Rojo × Región_Roja = Valor_R
# Canal Verde: Filtro_Verde × Región_Verde = Valor_G
# Canal Azul: Filtro_Azul × Región_Azul = Valor_B

### Resultado final:
Resultado = Valor_R + Valor_G + Valor_B

# Entrada: Imagen 3D (con colores)
# Salida: Imagen 2D (un solo canal)
# El filtro aprende a combinar información de los 3 colores para detectar patrones específicos

# Una capa convolucional puede tener varios filtros, cada uno de los cuales devuelve una imagen bidimensional que se puede reconvertir a una tridimensional. 

# Imagina que tienes 3 filtros diferentes en una capa:
# Filtro 1: Detecta bordes horizontales
# Filtro 2: Detecta bordes verticales
# Filtro 3: Detecta esquinas

# Cada filtro produce su propio mapa de características:
# Imagen de entrada (32×32×3) → Filtro 1 → Mapa 1 (30×30)
# Imagen de entrada (32×32×3) → Filtro 2 → Mapa 2 (30×30)  
# Imagen de entrada (32×32×3) → Filtro 3 → Mapa 3 (30×30)

# Los 3 mapas se apilan para formar una nueva imagen 3D:
# Resultado = 30×30×3
#     Ancho: 30
#     Alto: 30
#     Profundidad: 3 (porque usamos 3 filtros)

# En la siguiente capa convolucional, la profundidad de los filtros será igual al número de filtros de la capa anterior.

# Ejemplo completo:

# Capa 1:
# Entrada: 32×32×3 (imagen RGB)
# Filtros: 5 filtros de 3×3×3
# Salida: 30×30×5

# Capa 2:
# Entrada: 30×30×5 (salida de capa 1)
# Filtros: 8 filtros de 3×3×5 ← ¡Profundidad = 5!
# Salida: 28×28×8

# ¿Por qué la profundidad debe coincidir?
# El filtro debe procesar todos los canales de la entrada:
# Si la entrada tiene 5 canales, el filtro necesita 5 capas para procesarlos todos
# Es como tener 5 lentes diferentes, una para cada canal

# Las capas convolucionales contienen menos parámetros que las capas completamente conectadas, lo que hace que sea más fácil entrenarlas.

#---------------------------- ¿Qué es una imagen a color? ----------------------------
# Una imagen a color tiene 3 canales:
# Canal Rojo (R): Intensidad del rojo en cada píxel
# Canal Verde (G): Intensidad del verde en cada píxel
# Canal Azul (B): Intensidad del azul en cada píxel

#---------------------------- ¿Qué es un filtro 3x3x3? ----------------------------
# Filtro = 3 × 3 × 3
# 3 píxeles de ancho
# 3 píxeles de alto
# 3 de profundidad (uno para cada canal de color)

#---------------------------- ¿Cómo se mueve el filtro? ----------------------------
### Movimiento horizontal y vertical:
# El filtro se desliza de izquierda a derecha y de arriba a abajo por la imagen.

### NO se mueve en profundidad:
# El filtro NO se mueve a través de los canales. Siempre procesa los 3 canales simultáneamente.

"Cada color tiene pesos diferentes"
# El filtro tiene pesos específicos para cada canal:

# Filtro para canal ROJO:
[[w1, w2, w3],
 [w4, w5, w6],
 [w7, w8, w9]]

# Filtro para canal VERDE:
[[w10, w11, w12],
 [w13, w14, w15],
 [w16, w17, w18]]

# Filtro para canal AZUL:
[[w19, w20, w21],
 [w22, w23, w24],
 [w25, w26, w27]]

"Las imágenes resultantes se doblan"
# Esto significa que los 3 resultados (uno por cada canal) se suman para crear un solo valor:
Resultado_final = Resultado_Rojo + Resultado_Verde + Resultado_Azul

#---------------------------- capa convolucional  vs  capa completamente conectada ----------------------------
# La capa convolucional recibe una imagen input que tiene un tamaño de 32x32x3. 
# El tamaño del filtro es 3x3x3. 
# Las dimensiones de la imagen de output son 30x30x1 (con un solo canal). 
# Esta capa convolucional tiene 27 parámetros (3·3·3).

# capa completamente conectada. 
# Si se convierte una imagen de 32x32x3 en una de 30x30
# Eentonces a cada pixel de output se le agrega una neurona
# Requeriría un total de 900 neuronas (30·30). 
# Cada neurona está conectada a todos los pixeles de la imagen input
# Así que el número total de parámetros sería 2 764 800 (32·32·3·900).

# Calcular una convolución entre:
# Imagen:2×3×2 (alto × ancho × canales)
# Filtro: 2×1×2
import numpy as np
# Imagen 2x3x2
image = np.array([
    [[1, 3], [1, 2], [2, 3]],
    [[3, 1], [5,3], [2,1]]
])
# Filtro 2x1x2
kernel = np.array([
    [[0.5,0.5]],
    [[1,-1]]
])
H, W, C = image.shape
kH, kW, kC = kernel.shape
# Tamaño salida
out_h = H - kH + 1
out_w = W - kW + 1
output = np.zeros((out_h, out_w))
for i in range(out_h):
    for j in range(out_w):
        region = image[i:i+kH, j:j+kW, :]
        output[i, j] = np.sum(region * kernel)
print(output)
# Resultado: [1, 2, 5]

#---------------------------- ¿Para qué sirven estos cálculos? ----------------------------
# 1. Diseño de arquitecturas de red:
#     Cuando diseñas una CNN, necesitas saber exactamente qué tamaño tendrá cada capa para:
#     Planificar cuánta memoria necesitarás
#     Asegurarte de que las dimensiones sean compatibles entre capas
#     Controlar cómo se reduce progresivamente el tamaño de la imagen

# 2. Control del flujo de información:
#     Los parámetros que calculaste (padding, stride, tamaño de filtro) te permiten:
#     Preservar información importante (con padding)
#     Reducir el tamaño computacional (con stride > 1)
#     Extraer características específicas (con diferentes tamaños de filtro)

#---------------------------- configuraciones de la capa convolucional ----------------------------
#---------------------------- Padding o relleno: ----------------------------
# Esta configuración coloca ceros en los bordes de la matriz (padding cero) 
# para que los pixeles más alejados participen en la convolución al menos la misma cantidad de veces que los pixeles centrales.
# Esto evita que se pierda información importante. 
# Los ceros agregados también participan en la convolución 
# el tamaño del padding determina el ancho del padding cero.

# ENTRADA (5x5):          KERNEL (2x2):       RESULTADO (4x4):
# [0 0 0 0 0]                [0 1]              [0  3  8  4]
# [0 0 1 2 0]            *   [2 3]          =   [9 19 25 10]
# [0 3 4 5 0]                                   [21 37 43 16]
# [0 6 7 8 0]                                   [6  7  8  0]
# [0 0 0 0 0]

# los 0 alrededor de la imagen son el padding, y el resultado es una imagen de 4x4 en lugar de 5x5 porque el kernel no puede aplicarse a los bordes sin padding.

# CÁLCULO PASO A PASO:
# Posición [0,0]: (0×0)+(0×1)+(0×2)+(0×3) = 0
# Posición [0,1]: (0×0)+(0×1)+(0×2)+(1×3) = 3
# Posición [0,2]: (0×0)+(1×1)+(0×2)+(2×3) = 8
# Posición [1,1]: (0×0)+(1×1)+(3×2)+(4×3) = 19

# FÓRMULA TAMAÑO OUTPUT:
# Input: 5×5, Kernel: 2×2, Padding: 0, Stride: 1
# Output = (5-2+0)/1 + 1 = 4×4

#---------------------------- Paso ("stride" o "striding" en inglés) ----------------------------
# desplaza el filtro en más de un pixel y genera una imagen output más pequeña.

# STRIDE = 1 (paso normal):
# Input 5x5:          Filtro 2x2 se mueve de 1 en 1:
# [1 2 3 4 5]         Posición 1: [1 2]  →  Resultado
# [6 7 8 9 0]                     [6 7]
# [1 2 3 4 5]         Posición 2: [2 3]  →  4x4
# [6 7 8 9 0]                     [7 8]
# [1 2 3 4 5]         Posición 3: [3 4]  →  output
#                                 [8 9]

# STRIDE = 2 (salta de 2 en 2):
# Input 5x5:          Filtro 2x2 se mueve de 2 en 2:
# [1 2 3 4 5]         Posición 1: [1 2]  →  Resultado
# [6 7 8 9 0]                     [6 7]
# [1 2 3 4 5]         Posición 2: [3 4]  →  2x2
# [6 7 8 9 0]                     [8 9]
# [1 2 3 4 5]         (se salta posiciones)  →  output

# STRIDE = 3 (salta de 3 en 3):
# Input 5x5:          Filtro 2x2 se mueve de 3 en 3:
# [1 2 3 4 5]         Solo Posición 1: [1 2]  →  1x1
# [6 7 8 9 0]                      [6 7]     →  output
# [1 2 3 4 5]         (muy pocos cálculos)
# [6 7 8 9 0]
# [1 2 3 4 5]

# TAMAÑO OUTPUT = (W - K + 2P) / S + 1

# Donde:
# W = ancho input
# K = tamaño kernel  
# P = padding
# S = stride (paso)

# Ejemplos:
# - Input 5x5, Kernel 2x2, P=0, S=1 → Output = (5-2+0)/1+1 = 4x4
# - Input 5x5, Kernel 2x2, P=0, S=2 → Output = (5-2+0)/2+1 = 2.5 → 2x2
# - Input 5x5, Kernel 2x2, P=0, S=3 → Output = (5-2+0)/3+1 = 2x2

# - Stride = 1: Movimiento normal, output más grande
# - Stride > 1: Salta posiciones, output más pequeño, menos cálculos

# calcular el tamaño del tensor de output para la capa convolucional
# N = tamaño de la imagen
# F = tamaño del filtro
# P = padding
# S = stride (paso)

import math
N = 11   # tamaño imagen
F = 3    # tamaño filtro
P = 4    # padding
S = 2    # stride
output = (N - F + 2*P) / S + 1
print("Output size:", int(output), "x", int(output))
# Respuesta final: 9 x 9

#---------------------------- capas convolucionales en Keras ----------------------------
keras.layers.Conv2D(filters, kernel_size, strides, padding, activation)

# filters: el número de filtros, que corresponde al tamaño del tensor de output.
# kernel_size: las dimensiones espaciales del filtro K. Todo filtro es un tensor de tamaño K×K×D
#              donde D es igual a la profundidad de la imagen de input.
# strides: un paso (o stride) determina cuán lejos se desplaza el filtro sobre la matriz de input. 
#          De forma predeterminada, está configurado en 1.
# padding: este parámetro define el ancho del padding cero.
#          Hay dos tipos de padding: valid y same. 
#          El tipo por defecto es valid y es igual a cero. 
#          Same establece el tamaño del padding automáticamente de modo que el ancho y la altura del tensor de output sea igual al ancho y la altura del tensor de input.
# activation: esta función se aplica inmediatamente después de la convolución. 
#             De manera predeterminada, este parámetro es None

# Para que los resultados de la capa convolucional sean compatibles con la capa completamente conectada
# debes conectar una nueva capa llamada Flatten que convierte al tensor multidimensional en unidimensional.

# Por ejemplo, una imagen de input de 32x32x3 atraviesa una capa convolucional.
# El output es un tensor de estas dimensiones: (None, 30, 30, 4).
# "None" significa que el tamaño del lote es desconocido
# tener 4 filtros define la última dimensión
# Luego la aplanamos con una capa Flatten.
# Todo lo que está dentro de cada lote se remezcla y queda listo para ser procesado por una capa completamente conectada

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
model = Sequential()
# este tensor mide (None, 32, 32, 3)
# la primera dimensión define diferentes objetos
# está configurada en None porque desconocemos el tamaño del lote
model.add(Conv2D(filters=4, kernel_size=(3, 3), input_shape=(32, 32, 3)))
# este tensor mide (None, 30, 30, 4)
model.add(Flatten())
# este tensor mide (None, 3600)
# donde 3600 = 30*30*4
model.add(Dense(...))

#---------------------------- capas convolucionales, que resuelven? ----------------------------
# Primero pensemos en capas totalmente conectadas (Dense)
# Si tienes una imagen de 64×64 píxeles = 4096 píxeles
# Si conectas esto a una capa densa de 100 neuronas = 409,600 pesos
# Y eso solo para una capa.

# Problemas:
#     Demasiados parámetros
#     Pierdes la estructura espacial
#     No aprovechas que los píxeles cercanos están relacionados

#---------------------------- Idea clave de las CNN (Convolutional Neural Networks) ----------------------------
# usan filtros pequeños que recorren la imagen buscando patrones.

# Dense: mira toda la imagen
# Convolución: mira pequeños pedazos

# Ejemplo:
# imagen 28x28
# filtro 3x3
# El filtro se mueve por toda la imagen.


#---------------------------- Qué es una convolución? ----------------------------
# Multiplicar un pequeño filtro por una región de la imagen y sumar el resultado.

# Imagen:
# 2 3 1
# 4 6 2
# 1 2 0

# Filtro (kernel):
# 1 0
# 0 1

# Tomamos una ventana de la imagen:
# 2 3
# 4 6

# Multiplicamos elemento a elemento:
# 2*1 + 3*0
# 4*0 + 6*1

# Sumamos:
# 2 + 6 = 8
# Ese 8 es un pixel del feature map.

#---------------------------- Qué es un Kernel (o filtro)? ----------------------------
# Un kernel es una matriz pequeña de pesos.

# Ejemplo 3x3:
# 1 0 -1
# 1 0 -1
# 1 0 -1

# El kernel detecta patrones.

#---------------------------- Cómo se mueve el kernel (stride)? ----------------------------
# El stride es cuánto se mueve el filtro.

# Stride = 1
# posición 1
# posición 2
# posición 3

# Stride = 2
# Salta posiciones.

# Stride grande → reduce tamaño de la imagen

#---------------------------- Qué es el Feature Map? ----------------------------
# El resultado de aplicar un filtro es un = Feature Map

# Imagen → filtro bordes → resultado:
# 0 2 5 1
# 1 3 4 0

# Ese mapa indica = dónde existe ese patrón

# Ejemplo:
# Filtro detecta bordes verticales
# Feature map:
# 0 0 5 6
# 0 0 7 8
# Significa:
# hay bordes aquí

#---------------------------- Muchos filtros = muchas características ----------------------------
# Una capa convolucional no tiene solo un filtro.
# Puede tener:
# 32 filtros
# 64 filtros
# 128 filtros
# Cada uno detecta algo distinto.

#---------------------------- Qué es padding? ----------------------------
# Cuando aplicas convolución la imagen se hace más pequeña.

# Ejemplo:
# Imagen:
# 5x5
# Filtro:
# 3x3
# Resultado:
# 3x3
# Porque el filtro no cabe en los bordes.

# Solución: padding, agregas ceros alrededor.

#---------------------------- Flujo típico de una CNN ----------------------------
# imagen
#  ↓
# convolución
#  ↓
# ReLU
#  ↓
# pooling
#  ↓
# convolución
#  ↓
# ReLU
#  ↓
# pooling
#  ↓
# flatten
#  ↓
# dense
#  ↓
# output

#---------------------------- Qué es Pooling? ----------------------------
# Pooling reduce el tamaño del feature map(el número de los parámetros del modelo).
# El más común = Max Pooling

# Ejemplo:
# 2 6
# 3 1

# Resultado: 6

# Solo se queda con el máximo.

# Esto ayuda a:
#     Reducir tamaño
#     Mantener lo importante
#     Evitar overfitting

# Max Pooling:
    # Se determina el tamaño del kernel (por ejemplo, 2x2).
    # El kernel comienza a moverse de izquierda a derecha y de arriba a abajo; en cada cuadro de cuatro pixeles hay un pixel con el valor máximo.
    # El pixel con el máximo valor se mantiene, mientras que los que le rodean desaparecen.
    # El resultado es una matriz formada solo por los pixeles con los valores máximos.

# MaxPooling devuelve el valor máximo de pixel del grupo de pixel dentro de un canal. 
# Si la imagen de input tiene un tamaño de W×W, entonces el tamaño de la imagen de output es W/K, donde K es el tamaño del kernel

# AveragePooling:
#     devuelve el valor promedio de un grupo de pixeles dentro de un canal

keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')

#     pool_size: tamaño de pooling (agrupación). Cuanto más grande sea, más pixeles vecinos se involucran.
#     strides: un paso (o stride) determina cuán lejos se desplaza el filtro sobre la matriz de input. Si se especifica None, el paso es igual al tamaño de pooling.
#     padding: este parámetro define el ancho del padding cero. El tipo predeterminado del padding es valid, que es igual a cero. Same establece el tamaño del padding automáticamente.

# Los parámetros de MaxPooling2D son similares a estos.

#---------------------------- Qué detecta cada capa? ----------------------------
# Primera capa: bordes
# Segunda capa: formas simples
# Tercera capa: partes de objetos
# Capas profundas: objetos completos

#---------------------------- Por qué funcionan tan bien ----------------------------
# 1. Comparten pesos
#     Un filtro se usa en toda la imagen.
#     3x3 = 9 pesos

# 2. Detectan patrones en cualquier lugar

# 3. Preservan la estructura espacial
#     Las CNN entienden geometría de la imagen.

#---------------------------- Ejemplo de red convolucional con conectada ----------------------------
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import numpy as np
features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')
# Cambia las dimensiones de la imagen y establece el rango de [0, 1].
# Al cambiar el tamaño, puedes configurar una de las dimensiones a -1
# para que se calcule automáticamente
features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
features_test = features_test.reshape(-1, 28, 28, 1) / 255.0
model = Sequential()
model.add(
    Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1),
    )
)
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc']
)
model.summary() # muestra el número de parámetros en cada capa

# el tamaño de la imagen permanezca igual después de la primera capa pero disminuye a la mitad después de la segunda.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import numpy as np
features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')
features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
features_test = features_test.reshape(-1, 28, 28, 1) / 255.0
model = Sequential()
model.add(
    Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1),
        padding='same'
        
    )
)
model.add(
    Conv2D(
        filters=4,
        kernel_size=(3, 3),
        strides=2,
        padding='same',
        activation="relu",
    )
)
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc']
)
model.summary()
model.fit(
    features_train,
    target_train,
    epochs=1,
    verbose=1,
    steps_per_epoch=1,
    batch_size=1,
)

# Resultado
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d_2 (Conv2D)           (None, 28, 28, 4)         40

#  conv2d_3 (Conv2D)           (None, 14, 14, 4)         148

#  flatten_1 (Flatten)         (None, 784)               0

#  dense_1 (Dense)             (None, 10)                7850

# =================================================================
# Total params: 8,038
# Trainable params: 8,038
# Non-trainable params: 0
# _________________________________________________________________

# 1/1 [==============================] - ETA: 0s - loss: 2.4489 - acc: 0.0000e+00
# 1/1 [==============================] - 0s 421ms/step - loss: 2.4489 - acc: 0.0000e+00

#---------------------------- Arquitectura LeNet ----------------------------
# arquitectura popular para clasificar imágenes con tamaño de 20-30 pixeles

# 1. Primero, una imagen atraviesa dos veces un par de capas convolucionales y de pooling para que disminuya el tamaño del cuadrado: 32x32→ 28x28 → 14x14 → 10x10 → 5x5. 
#    el número de capas aumenta de 1 a 16. Se obtiene un tensor de 16 capas de 5x5.
#    Al final, este tensor se aplana para dar nodos de 5x5x16=400.

# 2. La fila de características aplanada se procesa mediante varias capas completamente conectadas
#    que por lo general tienen 120 → 84 → 10 neuronas. 
#    La última fila de 10 valores predice el dígito inicial.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')
features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
features_test = features_test.reshape(-1, 28, 28, 1) / 255.0
model = Sequential()
model.add(
    Conv2D(
        filters=4,
        kernel_size=(3, 3),
        padding='same',
        activation="relu",
        input_shape=(28, 28, 1),
    )
)
model.add(
    Conv2D(
        filters=4,
        kernel_size=(3, 3),
        strides=2,
        padding='same',
        activation="relu",
    )
)
model.add(MaxPooling2D(pool_size=(2,2))) #Max Pooling con un tamaño de 2x2 
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc']
)
model.summary()
model.fit(
    features_train,
    target_train,
    epochs=1,
    verbose=1,
    steps_per_epoch=1,
    batch_size=1,
)

# Resultado
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 4)         40        
                                                                 
#  conv2d_1 (Conv2D)           (None, 14, 14, 4)         148       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 7, 7, 4)          0         
#  )                                                               
                                                                 
#  flatten (Flatten)           (None, 196)               0         
                                                                 
#  dense (Dense)               (None, 10)                1970      
                                                                 
# =================================================================
# Total params: 2,158
# Trainable params: 2,158
# Non-trainable params: 0
# _________________________________________________________________

# 1/1 [==============================] - ETA: 0s - loss: 2.0844 - acc: 0.0000e+00
# 1/1 [==============================] - 2s 2s/step - loss: 2.0844 - acc: 0.0000e+00

# Solo relu puede superar a 'tanh'

# Flujo de datos:
# Imagen 28×28×1 
#     ↓ Conv2D (6 filtros)
# 28×28×6 
#     ↓ AvgPool2D
# 14×14×6 
#     ↓ Conv2D (16 filtros)
# 10×10×16 
#     ↓ AvgPool2D
# 5×5×16 
#     ↓ Flatten
# 400 
#     ↓ Dense (120)
# 120 
#     ↓ Dense (84)
# 84 
#     ↓ Dense (10)
# 10 (predicción final)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
import matplotlib.pyplot as plt
import numpy as np
features_train = np.load('/datasets/fashion_mnist/train_features.npy')
target_train = np.load('/datasets/fashion_mnist/train_target.npy')
features_test = np.load('/datasets/fashion_mnist/test_features.npy')
target_test = np.load('/datasets/fashion_mnist/test_target.npy')
features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
features_test = features_test.reshape(-1, 28, 28, 1) / 255.0
# LAS 8 CAPAS DE LeNet
model = Sequential()
# Capa 1: Conv2D
model.add(Conv2D(6, (5, 5), padding='same', activation='tanh',
                input_shape=(28, 28, 1)))
# Input: (28,28,1) → Output: (28,28,6)
# Capa 2: AveragePooling2D / Reduce dimensiones a la mitad
model.add(AvgPool2D(pool_size=(2, 2)))
# Input: (28,28,6) → Output: (14,14,6)
# Capa 3: Conv2D
model.add(Conv2D(16, (5, 5), padding='valid', activation='tanh'))
# Input: (14,14,6) → Output: (10,10,16)
# Capa 4: AveragePooling2D / Reduce dimensiones nuevamente
model.add(AvgPool2D(pool_size=(2, 2))) 
# Input: (10,10,16) → Output: (5,5,16)
# Capa 5: Flatten / Convierte matriz 2D en vector 1D
model.add(Flatten()) 
# Input: (5,5,16) → Output: (400)
# Capa 6: Dense
model.add(Dense(units=120, activation='tanh'))
# Input: (400) → Output: (120)
# Capa 7: Dense
model.add(Dense(units=84, activation='tanh'))
# Input: (120) → Output: (84)
# Capa 8: Dense (salida)
model.add(Dense(units=10, activation='softmax'))
# Input: (84) → Output: (10)
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['acc']
)
model.summary()
model.fit(
    features_train,
    target_train,
    epochs=1,
    verbose=1,
    steps_per_epoch=1,
    batch_size=1,
)
#---------------------------- El algoritmo Adam ----------------------------
# El descenso de gradiente estocástico (SGD) es un algoritmo de optimización 
# se utiliza para minimizar la función de pérdida en modelos de aprendizaje automático
# pero no es el algoritmo más eficiente para entrenar una red neuronal

# Si el paso es muy pequeño, el entrenamiento podría tardar demasiado. 
# Si es demasiado grande, puede que no logre el entrenamiento mínimo requerido.

# El algoritmo Adam automatiza la selección del paso. 
# Escoge diferentes parámetros para diferentes neuronas, lo que acelera el entrenamiento del modelo.
# Adam es la forma más rápida de encontrar el mínimo.

# Adam (Adaptive Moment Estimation) es un algoritmo de optimización más avanzado que combina las ventajas de dos métodos anteriores: AdaGrad y RMSProp.

# Método 1 - Flexibilidad de configuración:
from tensorflow.keras.optimizers import Adam
optimizer = Adam()
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['acc'],
)

#Método 2 - Usa valores por defecto:
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc']
)

# El principal hiperparámetro que se puede configurar en el algoritmo Adam es la tasa de aprendizaje.
# Esta es la parte del descenso de gradiente donde comienza el algoritmo
# optimizer = Adam(learning_rate=0.01)
# La tasa de aprendizaje predeterminada es 0.001. A veces reducirla puede ralentizar el aprendizaje, pero eso mejora la calidad del modelo en general.

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
def load_data():
    (features_train, target_train), (features_test, target_test) = fashion_mnist.load_data()
    features_train = features_train.reshape(-1, 28, 28, 1) / 255.0
    features_test = features_test.reshape(-1, 28, 28, 1) / 255.0
    return (features_train, target_train), (features_test, target_test)
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer= optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc'],
    )
    return model
def train_model(
        model,
        train_data,
        test_data,
        batch_size=32,
        epochs=5,
        steps_per_epoch=None,
        validation_steps=None,
):
    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(
        features_train,
        target_train,
        validation_data=(features_test, target_test),
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True,
    )
    return model
train_data, test_data = load_data()
input_shape = (28, 28, 1)
model = create_model(input_shape)
model_fitted = train_model(model,
                           train_data,
                           test_data,
                           epochs=2
                           )
# resultado:
# Epoch 1/2
# 1875/1875 - 6s - 3ms/step - acc: 0.7014 - loss: 0.8377 - val_acc: 0.7700 - val_loss: 0.6284
# Epoch 2/2
# 1875/1875 - 6s - 3ms/step - acc: 0.7865 - loss: 0.5796 - val_acc: 0.7978 - val_loss: 0.5622

#---------------------------- generadores de datos ----------------------------
# otro ejercicio de clasificación

# En el conjunto de datos hay varias carpetas con imágenes de estas frutas:
# Banana  Carambola  Mango
# Naranja Durazno    Pera
# Caqui   Pitaya     Ciruela
# Pomelo  Tomates    Melón

# En cada carpeta hay fotos de la clase correspondiente. Por ejemplo:
image = Image.open('/datasets/fruits_small/Banana/Banana027.png')
plt.imshow(np.array(image))
image = Image.open('/datasets/fruits_small/Persimmon/Persimmon33.png')
plt.imshow(np.array(image))

model.fit(features_train, target_train)
# features_train: una matriz de imagen NumPy de cuatro dimensiones
# target_train: contiene una matriz unidimensional con etiquetas de clase

# Las matrices se almacenan en la RAM, no en el disco duro del PC
# Para lidiar con una cantidad tan grande de imágenes, necesitas implementar la carga de datos dinámica.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#     forma lotes con imágenes y etiquetas de clase de acuerdo con las fotos que están en las carpetas.
# https://keras.io/api/data_loading/image/

# generador de datos
datagen = ImageDataGenerator()
# Para hacer que extraiga datos de una carpeta
datagen_flow = datagen.flow_from_directory(
    # carpeta con el conjunto de datos
    '/datasets/fruits_small/',
    # tamaño de la imagen objetivo
    target_size=(150, 150),
    # tamaño del lote
    batch_size=16,
    # modo de clase
    class_mode='sparse',
    # configurar un generador de números aleatorios
    seed=12345,
)
# resultado: Se encontraron 1683 imágenes pertenecientes a 12 clases.
# El generador de datos encontró 12 clases (carpetas) y un total de 1683 imágenes.

# Argumentos:
target_size=(150, 150)
# ancho y alto objetivo de la imagen. 
# Las carpetas pueden contener imágenes de distintos tamaños pero las redes neuronales necesitan que todas sean de las mismas dimensiones

batch_size=16
# el número de imágenes en los lotes
# Cuantas más imágenes haya, más eficaz será el entrenamiento del modelo.
# No cabrán demasiadas imágenes en la memoria de la GPU, así que 16 es el valor perfecto para iniciar.

сlass_mode='sparse'
# indica el modo de output de la etiqueta de clase
# sparse significa que las etiquetas corresponderán al número de la carpeta

# saber cómo se relacionan los números de clase con los nombres de las carpetas:
# índices de clase
print(datagen_flow.class_indices)
{'Banana': 0, 'Carambola': 1, 'Mango': 2, 'Naranja': 3, 'Durazno': 4, 'Pera': 5, 'Caqui': 6, 'Pitaya': 7, 'Ciruela': 8, 'Pomelo': 9, 'Tomates': 10, 'Melón': 11}

# devolverá un objeto a partir del cual se pueden obtener los pares "imagen-etiqueta" 
features, target = next(datagen_flow)
print(features.shape)
resultado: (16, 150, 150, 3)
# El resultado es un tensor de cuatro dimensiones con dieciséis imágenes de 150x150 y tres canales de colores.

# entrenar al modelo con estos datos, pasar el objeto datagen_flow al método fit()
# La época no debería ser demasiado larga
# Para limitar el tiempo de entrenamiento, hay que especificar el número de lotes de conjuntos de datos: steps_per_epoch

model.fit(datagen_flow, steps_per_epoch=len(datagen_flow))
# fit() debe contener conjuntos de entrenamiento y validación
# hay que crear dos generadores de datos para cada conjunto.
# indica que el conjunto de validación contiene
# 25% de objetos aleatorios
datagen = ImageDataGenerator(validation_split=0.25)
train_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    # indica que este es el generador de datos para el conjunto de entrenamiento
    subset='training',
    seed=12345,
)
val_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    # indica que este es el generador de datos para el conjunto de validación
    subset='validation',
    seed=12345,
)

# se inicia el entrenamiento así:
model.fit(
    train_datagen_flow,
    validation_data=val_datagen_flow,
    steps_per_epoch=len(train_datagen_flow),
    validation_steps=len(val_datagen_flow),
)

# MÉTODO TRADICIONAL (matrices en RAM):
# features_train - Matriz de 4 dimensiones:
features_train.shape = (60000, 28, 28, 1)

# Cada dimensión significa:
# Primera dimensión (60000): Número de imágenes
# Segunda dimensión (28): Alto de cada imagen
# Tercera dimensión (28): Ancho de cada imagen
# Cuarta dimensión (1): Canales de color (1=escala de grises, 3=RGB)

# Visualización de la estructura:
# features_train = [
#     imagen_1: [[pixel, pixel, pixel, ...],    ← 28 píxeles
#                [pixel, pixel, pixel, ...],    ← 28 píxeles  
#                [pixel, pixel, pixel, ...],    ← 28 filas
#                ...],
    
#     imagen_2: [[pixel, pixel, pixel, ...],
#                [pixel, pixel, pixel, ...],
#                ...],
    
#     imagen_3: [[pixel, pixel, pixel, ...],
#                ...],
    
#     ... (60000 imágenes en total)
# ]

# target_train - Matriz unidimensional:
target_train.shape = (60000,)
target_train = [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, ...]
# cada numero representa la etiqueta de clase correspondiente a cada imagen en features_train.

# PROBLEMA DE MEMORIA RAM:
# 60000 imágenes × 28 × 28 × 1 canal × 4 bytes = 188 MB
# 1,000,000 imágenes × 224 × 224 × 3 canales × 4 bytes = 600 GB 

SOLUCIÓN: GENERADORES
# ANTES (todo en RAM):
features_train = np.array([imagen1, imagen2, ..., imagen60000])  # 188 MB
target_train = np.array([5, 0, 4, ..., 9])                      # 240 KB
model.fit(features_train, target_train)  # Todo cargado

# AHORA (carga dinámica), carga solo 16 imágenes a la vez:
datagen_flow = datagen.flow_from_directory(...)  # Solo metadatos
features, target = next(datagen_flow)  # Solo 16 imágenes
print(features.shape)  # (16, 150, 150, 3) ← Solo 2.7 MB

# ESTRUCTURA DE CARPETAS:
# fruits/
# ├── train/
# │   ├── apple/          ← Carpeta = Clase "apple"
# │   │   ├── img1.jpg
# │   │   ├── img2.jpg
# │   │   └── img3.jpg
# │   ├── banana/         ← Carpeta = Clase "banana"  
# │   │   ├── img4.jpg
# │   │   ├── img5.jpg
# │   │   └── img6.jpg
# │   └── orange/         ← Carpeta = Clase "orange"
# │       ├── img7.jpg
# │       └── img8.jpg

# 1. ImageDataGenerator "lee" las carpetas:
datagen = ImageDataGenerator()
flow = datagen.flow_from_directory(
    'fruits/train/',
    batch_size=4,        # ← Quiero lotes de 4 imágenes
    target_size=(150,150)
)

# 2. Mapeo automático carpeta → número:
class_indices = {
    'apple': 0,    # ← Carpeta "apple" = clase 0
    'banana': 1,   # ← Carpeta "banana" = clase 1  
    'orange': 2    # ← Carpeta "orange" = clase 2
}

# 3. Formación de lotes:
# Lote 1 (4 imágenes aleatorias):
features = [
    img2.jpg,  # de carpeta apple/  
    img5.jpg,  # de carpeta banana/
    img1.jpg,  # de carpeta apple/
    img7.jpg   # de carpeta orange/
]
# automáticamente crea las etiquetas:
labels = [0, 1, 0, 2]  # apple, banana, apple, orange

# Resultado del lote:
batch_features.shape = (4, 150, 150, 3)  # 4 imágenes redimensionadas
batch_labels.shape = (4,)                # [0, 1, 0, 2]

# Cuando escribes target_size=(150, 150), los números significan:
# Primer número (150): altura de la imagen en píxeles
# Segundo número (150): ancho de la imagen en píxeles
# Si tus imágenes originales son de diferentes tamaños, se redimensionarán a 150x150 píxeles antes de ser procesadas por la red neuronal.

# ¿Qué es un lote?
#     Un lote (o "batch" en inglés) es un grupo de imágenes que se procesan juntas al mismo tiempo.
#     el modelo toma un grupo pequeño de imágenes y las procesa todas juntas. 
# ¿Cómo sé cuántas imágenes hay en un lote?
#     la cantidad se define con el parámetro batch_size

# ¿QUÉ HACE ImageDataGenerator si el lote esta incompleto?
# Opción 1: Lote incompleto (por defecto)
# Opción 2: Descartar lote incompleto
# - No hay problema si el último lote tiene menos imágenes
# - El modelo se adapta al tamaño del lote

# ¿Por qué usar GPU para imágenes?
# Las imágenes son matrices de números (píxeles), y las GPUs pueden procesar miles de operaciones matemáticas simultáneamente
# Las operaciones de convolución requieren muchos cálculos matriciales que las GPUs manejan mucho mejor que las CPUs

# ¿Qué hace class_mode='sparse'?
# en flow_from_directory() es muy importante porque determina cómo se formatean las etiquetas de clase que el generador produce.
# Las etiquetas se representan como números enteros (0, 1, 2, 3...)
# Cada imagen tiene una sola etiqueta que es un número

# Con class_mode='sparse', las etiquetas serían:
# Una imagen de manzana → etiqueta = 0
# Una imagen de naranja → etiqueta = 1
# Una imagen de plátano → etiqueta = 2

# ¿Qué devuelve flow_from_directory()?
# train_datagen_flow = datagen.flow_from_directory(...)
# NO obtienes las imágenes directamente. 
# En su lugar, obtienes un objeto generador que puede producir datos bajo demanda.

# ¿Cómo funciona la función next()?
# La función next() le dice al generador: "Dame el siguiente lote de datos"
# Primer llamado a next()
batch_1_images, batch_1_labels = next(train_datagen_flow)
# batch_1_images: array con 16 imágenes (16, 150, 150, 3)
# batch_1_labels: array con 16 etiquetas [0, 1, 2, 0, 1, ...]

# Segundo llamado a next()  
batch_2_images, batch_2_labels = next(train_datagen_flow)
# Obtiene las SIGUIENTES 16 imágenes y etiquetas

# por qué hablamos de 4 dimensiones cuando tenemos un lote de imágenes?
# los datos se organizan en lo que llamamos un tensor de 4 dimensiones.

# Las 4 dimensiones son:
#     Batch size (lote): Número de imágenes que procesamos juntas = 16
#     Height (altura): Altura de cada imagen = 150 píxeles
#     Width (ancho): Ancho de cada imagen = 150 píxeles
#     Channels (canales): Canales de color = 3 (RGB)
# Formato del tensor: (batch_size, height, width, channels)
# En tu caso: (16, 150, 150, 3)

#----------------------------  ----------------------------
# ¿Qué hace steps_per_epoch?
# controla cuántos lotes (batches) procesa el modelo en cada época.
# Normalmente, una época significa procesar todo el conjunto de datos una vez. 
# Pero con generadores, necesitamos decirle al modelo cuántos pasos dar.

# Ejemplo práctico:

# Supongamos que tienes:
# 1000 imágenes en total
# batch_size = 16 (procesas 16 imágenes a la vez)

# Sin steps_per_epoch:
# Una época completa = 1000 ÷ 16 = 62.5 pasos (se redondea a 63)
# El modelo procesaría todas las 1000 imágenes

# Con steps_per_epoch=1:
# Una época = solo 1 paso
# El modelo procesaría solo 16 imágenes (un lote)

# en lugar de leer un libro completo (época normal), solo lees la primera página (steps_per_epoch=1) para verificar que puedes leer.

# Ejemplo de uso:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

datagen = ImageDataGenerator(validation_split=0.25)
train_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    # indica que este es el generador de datos para el conjunto de entrenamiento
    subset='training',
    seed=12345,
)
val_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    # indica que este es el generador de datos para el conjunto de validación
    subset='validation',
    seed=12345,
)
model = Sequential()
model.add(
    Conv2D(
        filters=6,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(150, 150, 3),
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=12, activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']
)
model.fit(train_datagen_flow,
          validation_data=val_datagen_flow,
          steps_per_epoch=1,
          validation_steps=1,
          verbose=2, epochs=1)

# rescale es responsable de la normalización.
# rescale=1./255 convierte los valores de píxeles de [0, 255] a [0, 1], lo que ayuda a que el modelo aprenda más rápido y con mayor precisión.
# Esta operación divide cada píxel por 255. Cada píxel se multiplica por 0.00392, lo que equivale a dividir por 255.

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255 )
train_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    seed=12345)
val_datagen_flow = datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='validation',
    seed=12345)
features, target = next(train_datagen_flow)
print(features[0])

# Resultado: 
# Found 1266 images belonging to 12 classes.
# Found 417 images belonging to 12 classes.
# [[[0.4784314  0.454902   0.454902  ]
#   [0.48235297 0.50980395 0.4901961 ]
#   [0.45882356 0.48627454 0.4666667 ]
#   ...
#   [0.21960786 0.24705884 0.2392157 ]
#   [0.2627451  0.28627452 0.27450982]
#   [0.30980393 0.32941177 0.32156864]]
#  [[0.4666667  0.48627454 0.43921572]
#   [0.50980395 0.48627454 0.48627454]
#   [0.48235297 0.50980395 0.4901961 ]
#   ...
#   [0.21176472 0.2392157  0.23137257]
#   [0.24313727 0.25490198 0.25882354]
#   [0.2901961  0.29803923 0.29411766]]
#  [[0.4666667  0.48627454 0.43921572]
#   [0.4784314  0.454902   0.454902  ]
#   [0.48235297 0.50980395 0.4901961 ]
#   ...
#   [0.20000002 0.21960786 0.21960786]
#   [0.2392157  0.24705884 0.25490198]
#   [0.25882354 0.26666668 0.27058825]]
#  ...
#  [[0.5686275  0.5921569  0.50980395]
#   [0.6039216  0.627451   0.5411765 ]
#   [0.6509804  0.67058825 0.57254905]
#   ...
#   [0.0627451  0.11764707 0.09019608]
#   [0.0627451  0.11764707 0.09019608]
#   [0.06666667 0.12156864 0.09411766]]
#  [[0.6509804  0.67058825 0.57254905]
#   [0.6784314  0.69803923 0.6       ]
#   [0.6862745  0.69803923 0.59607846]
#   ...
#   [0.05882353 0.10980393 0.08627451]
#   [0.0627451  0.11764707 0.09019608]
#   [0.0627451  0.11764707 0.09019608]]
#  [[0.6862745  0.69803923 0.59607846]
#   [0.7294118  0.7372549  0.63529414]
#   [0.7843138  0.7843138  0.6745098 ]
#   ...
#   [0.0509804  0.10588236 0.07843138]
#   [0.0627451  0.11764707 0.09019608]
#   [0.0627451  0.11764707 0.09019608]]]

#---------------------------- Aumento de datos de imágenes ----------------------------
# Si hay muy pocas muestras de entrenamiento, la red podría sobreentrenarse.
# Para evitar este problema, se utiliza el aumento.

# El aumento se usa para expandir artificialmente un conjunto de datos al transformar las imágenes existentes.
# Los cambios solo se aplican a los conjuntos de entrenamiento, mientras que los de prueba y validación se quedan igual.
# El aumento transforma la imagen original pero aún preserva sus características principales 
# (rotar, voltear, cambiar el brillo/contraste, Estirar y comprimir, Difuminar y enfocar, agregar ruido.) 
# para que el modelo pueda aprender a reconocer la clase incluso con variaciones.
# Esto es útil porque no cambiará las características fundamentales del objeto

# el aumento puede causar problemas
# la clase de la imagen puede cambiar o el resultado podría terminar como una pintura impresionista debido a las alteraciones
# Esto afecta la calidad del modelo.

# - No hagas aumento en los conjuntos de prueba y validación para evitar distorsionar los valores de las métricas.
# - Agrega aumentos de forma gradual, uno a la vez, y pon atención a la métrica de calidad del conjunto de validación.
# - Siempre deja sin cambiar algunas imágenes del conjunto de datos.

# cada tipo de aumento debe adaptarse a tu conjunto de datos específico.
# Si estás trabajando con imágenes de dígitos escritos a mano, no tiene sentido rotar las imágenes, pero sí podría ser útil agregar ruido o cambiar el brillo.
# Si estás trabajando con imágenes de objetos cotidianos, rotar o voltear las imágenes podría ser beneficioso para que el modelo aprenda a reconocer los objetos desde diferentes ángulos.

#---------------------------- Aumentos en keras ----------------------------
# En ImageDataGenerator hay muchas maneras de agregar aumentos de imagen. Por defecto, están deshabilitadas.

# giro vertical:
datagen = ImageDataGenerator(validation_split=0.25,
                             rescale=1./255,
                             vertical_flip=True)

# Deben crearse diferentes generadores para los conjuntos de entrenamiento y validación:
train_datagen = ImageDataGenerator(
    validation_split=0.25, rescale=1.0 / 255, horizontal_flip=True
)
validation_datagen = ImageDataGenerator( #aqui no se aplica el aumento
    validation_split=0.25, rescale=1.0 / 255
)
train_datagen_flow = train_datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    seed=12345,
)
val_datagen_flow = validation_datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='validation',
    seed=12345,
)
# Configura los objetos train_datagen_flow y val_datagen_flow al mismo valor de seed para evitar que estos conjuntos compartan elementos.

# Argumentos responsables del aumento:

# Reflejos (Booleanos)
# horizontal_flip y vertical_flip: Usan valores booleanos (True/False)
horizontal_flip=True   # Activa el reflejo horizontal
vertical_flip=True     # Activa el reflejo vertical

Rotaciones (Enteros)
# rotation_range: Usa un número entero que representa grados
rotation_range=90      # Rotaciones aleatorias de 0 a 90 grados
rotation_range=45      # Rotaciones aleatorias de 0 a 45 grados

Desplazamientos (Decimales)
# width_shift_range y height_shift_range: Usan números decimales (float) que representan porcentajes
width_shift_range=0.2   # Desplazamiento horizontal hasta 20%
height_shift_range=0.2  # Desplazamiento vertical hasta 20%

# Ejemplo de uso:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
train_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=90)
validation_datagen = ImageDataGenerator(
    validation_split=0.25,
    rescale=1./255)
train_datagen_flow = train_datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='training',
    seed=12345)
val_datagen_flow = validation_datagen.flow_from_directory(
    '/datasets/fruits_small/',
    target_size=(150, 150),
    batch_size=16,
    class_mode='sparse',
    subset='validation',
    seed=12345)
features, target = next(train_datagen_flow)
fig = plt.figure(figsize=(10,10))
for i in range(16):
    fig.add_subplot(4, 4, i+1)
    plt.imshow(features[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

#---------------------------- Redes convolucionales para la clasificación de frutas ----------------------------
#---------------------------- Codigo dado por el profesor ----------------------------
#Librerias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parte 1. Carga de datos
def load_data(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.)
    data_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)
    return data_datagen_flow

# Paso 2. Creacion del modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])
    return model

# Paso 3. Entrenamiento del modelo
def train_model(model, train_data, test_data, batch_size=None, epochs=10):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              verbose=2)
    return model
input_shape = (150, 150, 3)
train, val = load_data('/datos/fruits_small/')
model = create_model(input_shape)
model = train_model(model, train, val)

#---------------------------- # Ejercicio propio: ----------------------------
# Red neuronal convolucional entrenada con el conjunto de datos de frutas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Parte 1. Carga de datos
def load_data(path):
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
    )
    validation_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
    )
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345
    )
    val_datagen_flow = validation_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='validation',
        seed=12345
    )
    return train_datagen_flow, val_datagen_flow

# Paso 2. Creación del modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(train_data.num_classes, activation='softmax')) #Automatico
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Paso 3. Entrenamiento
def train_model(model, train_data, val_data, epochs=10):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        steps_per_epoch=200,
        validation_steps=50
    )

# Ejecución
input_shape = (150,150,3)
train_data, val_data = load_data('datos/fruits_small')
model = create_model(input_shape)
train_model(model, train_data, val_data)
print("Clases:", train_data.class_indices)
print("Numero de clases:", train_data.num_classes)

#---------------------------- Arquitectura ResNet (red residual) ----------------------------
# La principal diferencia entre VGG y otras arquitecturas es que se dejan de lado los filtros grandes (como 9x9 u 11x11, por ejemplo) en la capa convolucional.
# Quienes crearon VGG demostraron que varias convoluciones de 3x3 pueden imitar kernels de convolución más grandes.
# Un par de convoluciones con un filtro de 3x3 cubre el mismo número de pixeles que uno de 5x5
# solo hay 18 parámetros en dos convoluciones de 3x3 ((2·(3·3)), mientras que una de 5x5 contiene 25 (5·5). 
# En ResNet, todas las convoluciones, a excepción de la primera, tienen filtros que no son mayores que 3x3.

# ¿Qué hace a ResNet diferente de sus predecesoras?
# Las conexiones especiales que funcionan como atajos para saltar algunas capas.
# Se les llama conexiones-salto.

# Las conexiones-salto funcionan muy bien porque mantienen el flujo de la señal incluso cuando atraviesa una red muy profunda. Y cuanto más profunda sea la red, más funciones complejas puede aprender.
# El bloque cuello de botella permite reducir el número de pesos
# Por ejemplo, si el input del modelo es una matriz tridimensional con 256 canales, entonces, en vez de un par de convoluciones con filtros de 3x3, la operación de convolución se hace con un filtro de 1x1. 
# Esto reduce cuatro veces el número de canales hasta 64. 
# A esto le sigue una convolución de 3x3 con el mismo número de canales y después, una convolución de 1x1 devuelve el número hasta 256.

# La convolución "pesada" con el filtro de 3x3 se creó con un tensor cuatro veces más pequeño que la original, lo que redujo el número de parámetros y aceleró los cálculos.
# ResNet es una red profunda que utiliza conexiones-salto, convoluciones pequeñas y bloques cuello de botella

#---------------------------- ResNet en Keras ----------------------------
from tensorflow.keras.applications.resnet import ResNet50
model = ResNet50(input_shape=None, 
                 classes=1000,
                 include_top=True,
                 weights='imagenet')

# input_shape: tamaño de la imagen de input
#     No es completamente automático. 
#     Significa que el modelo inferirá la forma de entrada basándose en los primeros datos que reciba.

# classes=1000: el número de neuronas en la última capa totalmente conectada donde sucede la clasificación
#     El modelo puede clasificar entre 1000 categorías diferentes. 

# weights='imagenet': la inicialización de los pesos.
#     ImageNet es el nombre de una gran base de datos de imágenes que se usaba al entrenar la red para que clasificara imágenes en 1000 clases.
#     Escribe weights=None para inicializar los pesos al azar.

# include_top=True: indica que hay dos capas al final de ResNet
#     (GlobalAveragePooling2D y Dense) 
#     Si la configuras en False, estas dos capas no estarán presentes.

# - GlobalAveragePooling2D funciona como ventana para todo el tensor. Como AveragePooling2D, devuelve el valor promedio de un grupo de pixeles dentro de un canal.
# - Dense es la capa totalmente conectada responsable de la clasificación.

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
backbone = ResNet50( # red troncal
    # ResNet50 es la parte de la red que se encarga de extraer características de las imágenes.
    input_shape=(150, 150, 3), 
    weights='imagenet', 
    include_top=False
)

# ¿Qué hace?
# Toma ResNet50 pre-entrenada en ImageNet
# Quita la parte superior (include_top=False)
# Solo mantiene las capas que extraen características

model = Sequential()
model.add(backbone) # Paso 1: Red pre-entrenada
model.add(GlobalAveragePooling2D()) # Paso 2: Aplanar datos
model.add(Dense(12, activation='softmax'))  # Paso 3: Clasificar

# ¿Por qué cada capa?
    # backbone: Extrae características complejas (bordes, formas, texturas)
    # GlobalAveragePooling2D: Convierte los mapas de características en un vector plano
    # Dense(12): Clasifica en 12 categorías

# tenemos que "congelar" una parte de la red: algunas capas se quedarán con pesos de ImageNet y no entrenarán con descenso de gradiente. 
# Vamos a entrenar una o dos capas completamente conectadas en la parte superior de la red. De esta manera, se reducirá el número de parámetros de la red, pero se conservará la arquitectura.

backbone = ResNet50(
    input_shape=(150, 150, 3), weights='imagenet', include_top=False
)

# congela ResNet50 sin la parte superior
backbone.trainable = False
model = Sequential()
model.add(backbone)
model.add(GlobalAveragePooling2D())
model.add(Dense(12, activation='softmax'))

# No congelamos la capa completamente conectada encima de *backbone* para que la red pueda aprender.

# Congelar la red te permite evitar el exceso de entrenamiento e incrementar la tasa de aprendizaje de la red (el descenso de gradiente no necesitará contar derivadas para las capas congeladas).

#---------------------------- ¿Qué problema estamos resolviendo? ----------------------------
# Queremos clasificar imágenes.

# Pero entrenar una red desde cero:
# requiere MUCHOS datos
# tarda mucho
# puede salir mal

# Transfer Learning (aprendizaje por transferencia)
# Usamos una red ya entrenada (con millones de imágenes) y la adaptamos a nuestro problema.

#---------------------------- ¿Qué es ResNet50? ----------------------------
# Es una red neuronal profunda (50 capas) diseñada para extraer características de imágenes.
# permet d’entraîner efficacement des réseaux très profonds sans perte de performance due au gradient évanescent.
# ResNet50 repose sur l’apprentissage résiduel, une technique où chaque bloc convolutif intègre une « connexion de saut » (skip connection) reliant directement son entrée à sa sortie.
# Ces raccourcis stabilisent la rétropropagation du gradient, autorisant des profondeurs bien supérieures à celles des réseaux classiques

# ¿Qué hace internamente?
# Convierte una imagen así: imagen (pixeles)
# en algo así: bordes → texturas → formas → objetos

# Es decir:
#     - Detecta líneas
#     - Luego patrones
#     - Luego partes de objetos
#     - Luego objetos completos

# Cada neurona = una categoría

# include_top=True
# Incluye la parte final del modelo:
#     - GlobalAveragePooling2D
#     - Dense (clasificador)
# Es decir: ResNet completa (feature extractor + clasificador)

# ¿Qué cambia con include_top=False?
# Quitamos la parte final (clasificador).
# Nos quedamos SOLO con: imagen → extracción de características

# Entonces backbone es: Un extractor de características

# Paso 1: model.add(backbone)
# La imagen entra y sale como:
# (150,150,3) → mapa de características (tensor grande)

# Paso 2: GlobalAveragePooling2D()
# Convierte esto:
#     tensor 3D (alto, ancho, canales)
# en:
#     vector 1D

# ¿Cómo?
# Promedia cada canal.

# Ejemplo:
# [feature map] → promedio → 1 número por canal

# Paso 3: Dense(12, softmax)
# Clasifica en 12 categorías:
# vector → probabilidades (12 clases)

# Congelar la red
# backbone.trainable = False

# ¿Qué significa?
# Que los pesos de ResNet:
# - NO se actualizan
# - NO se entrenan
# - NO cambian

# Ventajas:
# - Evitas overfitting
# - Necesitas menos datos
# - Entrenamiento más rápido
# - Aprovechas conocimiento previo

# ResNet ya sabe:
# - detectar bordes
# - detectar formas
# - detectar texturas

# No necesitas reentrenar eso
# Solo necesitas: "traducir esas features a tus clases"

# Para reducir el tiempo invertido en el entrenamiento del modelo, podemos hacer uso de modelos previamente entrenados, lo que nos ayudará a entrenar nuestro modelo en menos tiempo.

#---------------------------- ResNet en Keras hecho por mi----------------------------
# usar ResNet50 para entrenar una clasificación de imágenes de frutas.

# 0. Importar las librerías necesarias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50

# 1. cargar datos
def load_data(path, width, height, batch_size):
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
    )
    validation_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
    )
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        seed=12345,
        shuffle=True, #Evita que el modelo aprenda patrones del orden de los datos, cada epoca ve datos diferentes 
    )
    val_datagen_flow = validation_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        seed=12345,
    )
    return train_datagen_flow, val_datagen_flow

# 2. Crear el modelo() 
def create_resnet_model(input_shape):
    backbone = ResNet50(
        input_shape=input_shape, 
        weights='imagenet', 
        include_top=False
        )
    # usaremos un modelo previamente entrenado con sus parámetros aprendidos como funciones para entrenar nuestro nuevo modelo
    for layer in backbone.layers:
        layer.trainable = False
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D()) # Paso 2: Aplanar datos
    model.add(Dense(train_gen.num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Entrenar el modelo()
def train_model(model, train_data, val_data, epochs=10):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch = len(train_data),
        validation_steps = len(val_data)
    )
    return model

# 4. usar el modelo:
## Parámetros
width = 160
height = 128
batch_size = 16
input_shape = (width, height,3)
## Datos almacenados
path = 'dataset/fruits/'
# Carga los datos de entrenamiento y validación
train_gen, val_gen = load_data(path, width, height, batch_size)
# Crea y entrena el modelo simple
model = create_resnet_model(input_shape)
trained_model = train_model(model, train_gen, val_gen, epochs=10)

#---------------------------- solucion dad por la leccion ----------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def load_data(path, width, height, batch_size):
    train_datagen = ImageDataGenerator(rescale=1/255,
                                    horizontal_flip=True,
                                    vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(path,
                                                        seed=123,
                                                        target_size = (width,height),
                                                        batch_size = batch_size,
                                                        class_mode = "categorical",
                                                        shuffle=True,
                                                        )
    return train_generator

def create_resnet_model(input_shape):
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    # usaremos un modelo previamente entrenado con sus parámetros susceptibles de ser aprendidos intactos
    for layer in backbone.layers:
        layer.trainable = False
    # entrenemos la parte de clasificación
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    # también podrías agregar una capa dense aquí: model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data, val_data, epochs=10):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )
    return model

#---------------------------- Elegir la solución más eficaz ----------------------------
# No necesitas ningún tipo de sistema intrincado o elaborado cuando todo lo que necesitas es un algoritmo simple para resolver la tarea en cuestión.
# La solución más simple suele ser la mejor. 
# Al elegir una herramienta o una técnica, considera la necesidad de recursos y la facilidad de uso.

# Un restaurante necesita un modelo que estime los tiempos de entrega con base en la distancia a la dirección del cliente. 
# Tienes la siguiente información: historial de pedidos con direcciones de clientes, tiempos de pedido y tiempos de entrega reales. 

# Primera eleccion:
#     Calcular el tiempo de viaje usando Apple o Google Maps.
# Segunda eleccion: 
#     Entrenar un modelo con los datos existentes de envío, clima y tiempo de viaje de Apple o Google Maps.
# Demasiado complicado o inadecuado:
#     Calcular el tiempo máximo de entrega e indícalo siempre en la página de pedido del cliente.
#     Crear una red neuronal que calcule los tiempos de viaje utilizando imágenes satelitales de Apple o Google Maps.

# A un banco le gustaría acelerar el servicio al cliente al reducir el tiempo que los clientes dedican a ingresar su información en el sistema. 
# Primera eleccion:
#     Entrenar una red neuronal para que reconozca números en las identificaciones escaneadas.
# Segunda eleccion: 
#     Crear una red neuronal para capturar todos los datos de los escaneos de identificaciones.
# Demasiado complicado o inadecuado:
#     Utilizar la regresión logística para capturar los datos a partir del escaneo de identificaciones.
#     Analizar el texto de las identificaciones y usar combinaciones recurrentes de palabras y números para desarrollar teclados personalizados.

#---------------------------- analisis de la tarea ----------------------------
# Imagina que vas a probar la red ResNet50 que se entrenó previamente usando el conjunto de datos de ImageNet. 
# Tiene 1000 clases, por lo que hay 1000 neuronas en la capa de salida. 
# ¿Cuántas neuronas de la capa de salida necesitas para adivinar la edad de 1 persona?
# Solo necesitamos una neurona para devolver el valor de edad predicho. Por lo tanto, la capa de salida de 1000 neuronas debe eliminarse de la red y reemplazarse por una nueva.

# SoftMax: no funcionará para una neurona porque requiere más y normalmente se usa para tareas de clasificación multiclase.
# ReLU: o cambia los números positivos y lleva todos los negativos a cero. No puede haber números menores que cero

# ¿Qué funciones de pérdida serían adecuadas para esta tarea?
#     Error cuadrático medio: adecuada para problemas de regresión.
#     Error absoluto medio: adecuada para problemas de regresión.

#---------------------------- Análisis Exploratorio de Datos de Rostros ----------------------------
# Librerías para manejo de datos y visualización
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
# Configuración de estilo
sns.set(style="whitegrid") # Configuran la visualización de gráficos en tu notebook
# %matplotlib inline - Hace que los gráficos aparezcan directamente en el notebook

#Cargar etiquetas y explorar el dataset
# Ruta al CSV con las etiquetas
labels_path = '/datasets/faces/labels.csv'
# Cargar datos
labels_df = pd.read_csv(labels_path)
# Ver primeras filas
print(labels_df.head())
# Revisar tamaño del dataset
print("Número de muestras:", len(labels_df))
# Revisar columnas disponibles
print("Columnas:", labels_df.columns.tolist())
# Tipos de datos y valores nulos
print(labels_df.info())
print(labels_df.isnull().sum())

#Explorar la distribución de edades
# Histograma de edades
plt.figure(figsize=(10,5))
sns.histplot(labels_df['age'], bins=20, kde=True, color='skyblue')
plt.title("Distribución de edades en el dataset")
plt.xlabel("Edad")
plt.ylabel("Número de imágenes")
plt.show()
# Resumen estadístico
print(labels_df['age'].describe())

# - Con esto veríamos si el dataset está equilibrado (muchas edades concentradas) o sesgado

#Visualizar algunas imágenes por edad
# Carpeta con imágenes
images_path = Path('/datasets/faces/final_files/')

# Seleccionar 10-15 imágenes al azar o por edad específica
sample_df = labels_df.sample(15, random_state=12345)  # 15 imágenes aleatorias
plt.figure(figsize=(15,8))
for i, row in enumerate(sample_df.itertuples()):
    img_file = images_path / row.filename
    try:
        img = Image.open(img_file)
        plt.subplot(3, 5, i+1)
        plt.imshow(img)
        plt.title(f"Edad: {row.age}")
        plt.axis('off')
    except:
        print(f"No se pudo abrir la imagen {img_file}")
plt.tight_layout()
plt.show()

# - Esto nos permite tener una impresión visual del dataset, ver iluminación, ángulos, tamaños y diversidad de rostros.
    # Si la calidad del conjunto de validación está mejorando, pero el modelo se está sobreajustando, no te apresures a cambiar el modelo. 
    # No es raro que las redes neuronales con una gran cantidad de capas se sobreajusten mucho.
    # Es aceptable siempre y cuando sean buenas en el conjunto de validación.