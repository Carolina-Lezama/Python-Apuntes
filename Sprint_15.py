#---------------------------- Complejidad computacional ----------------------------
# El tiempo de ejecución de algoritmos no se mide en segundos. 
# Está determinado por el número de operaciones elementales que realiza el algoritmo: sumas, multiplicaciones y comparaciones.

# El tiempo de ejecución en un PC suele llamarse tiempo de ejecución real o tiempo de reloj de pared
# El tiempo de ejecución del algoritmo determinado por el tiempo transcurrido en el reloj(real).

# El tiempo de ejecución de un algoritmo también depende de sus argumentos

def increase(elements):
    n = len(elements)     # 2 operaciones, 1 para la llamada a len,  1 para asignar a n
    for i in range(n):    # 1 operación + 2 operaciones * n, 1 para la llamada a range + 2 por cada iteración (comparación y aumento de i)
        elements[i] += 1  # 2 operaciones (para cada valor de i), 1 para acceder a elements[i] + 1 para la suma

# Es imposible calcular el tiempo de ejecución en algoritmos complejos. Por eso determinamos su complejidad computacional, o tiempo de ejecución asintótico.

# El tiempo de trabajo es una función de n, escrita como T(n).
# El tiempo de ejecución asintótico de un algoritmo crece a medida que n se incrementa.

# 1) Si T(n) = 4n + 3, el tiempo de ejecución asintótico T(n) ~ n. El algoritmo tiene complejidad lineal.
# 2) Si T(n) = 5n² + 3n - 1, el tiempo de ejecución asintótico T(n) ~ n². El algoritmo tiene complejidad cuadrática.
# 3) Si T(n) = 10n³ - 2n² + 5, entonces T(n) ~ n³. El algoritmo tiene complejidad cúbica.
# 4) Si T(n) = 10, entonces T(n) ~ 1.El algoritmo tiene complejidad constante, es decir, no depende de n.

#---------------------------- Tiempo de entrenamiento de un modelo de regresión lineal ----------------------------
Fórmula de entrenamiento de regresión lineal:
w = (X^T * X)^(-1) * X^T * y

Paso 1: Multiplicación X^T * X

Tamaño de matrices: X es (n × p), X^T es (p × n)
Resultado: Matriz (p × p)
Complejidad: T(n,p) ~ np²
Paso 2: Inversión de matriz

Operación: Calcular (X^T * X)^(-1)
Tamaño: Matriz (p × p)
Complejidad: T(n,p) ~ p³
Paso 3: Multiplicación por X^T

Operación: (X^T * X)^(-1) * X^T
Resultado: Matriz (p × n)
Complejidad: T(n,p) ~ np²
Paso 4: Multiplicación final por y

Operación: Resultado anterior * y
Complejidad: T(n,p) ~ np
Complejidad total:
T(n,p) ~ np² + p³ + np² + np
Simplificando (cuando p < n):

T(n,p) ~ np²
Conceptos clave que aprendimos:
1. Notación de complejidad:
n: Número de observaciones
p: Número de características
T(n,p): Función de complejidad computacional
2. Regla de simplificación:
Cuando p < n (más observaciones que características), el término dominante es np².

3. Implicación práctica:
"Si hay muchas características, el entrenamiento llevará más tiempo"

#---------------------------- Metodos iterativos ----------------------------
estos no te darán una solución precisa, sino solo una aproximada, pero son mas rapidos. 
El algoritmo realiza iteraciones similares repetidamente y la solución se vuelve más precisa con cada paso. 
En el caso de que no se necesite una gran precisión, bastará con unas pocas iteraciones.

La complejidad computacional de los métodos iterativos depende del número de pasos realizados, que puede verse afectado por la cantidad de datos.

En cada iteración, el segmento con la raíz se divide entre 2. 
Una vez que se alcance una longitud de segmento inferior a e, el algoritmo podrá detenerse. 
Esta condición se denomina criterio de parada.

import math
def bisect(function, left, right, error):
    while right - left > error:
        if function(left) == 0:
            return left
        if function(right) == 0:
            return right
        middle = (left + right) / 2
        if function(left) * function(middle) < 0:
            right = middle
        else:
            left = middle
    return left
def f1(x):
    return x ** 3 - x ** 2 - 2 * x
def f2(x):
    return (x + 1) * math.log10(x) - x ** 0.75
print(bisect(f1, 1, 4, 0.000001))
print(bisect(f2, 1, 4, 0.000001))

Es un algoritmo iterativo para encontrar las raíces de una función (valores donde f(x) = 0).

#---------------------------- Métodos Directos ----------------------------
Son algoritmos que encuentran la solución exacta usando una fórmula o procedimiento determinado en un solo paso.

Características principales:
Precisión: Dan la solución exacta, no aproximada
Complejidad fija: Su tiempo de ejecución es independiente de los datos específicos
Predictibilidad: Siempre realizan la misma cantidad de operaciones
Ejemplo: La fórmula para regresión lineal que viste: W = (X^T × X)^(-1) × X^T × y

#---------------------------- Métodos Iterativos ----------------------------
Son algoritmos que encuentran la solución mediante pasos repetidos, mejorando la aproximación en cada iteración.

Características principales:
Precisión: Dan soluciones aproximadas (pero pueden ser muy precisas)
Complejidad variable: El tiempo depende del número de iteraciones necesarias
Flexibilidad: Puedes parar cuando tengas suficiente precisión
Ejemplo: El método de bisección, o el descenso de gradiente.

#---------------------------- Minimización de la función de pérdida ----------------------------
Esta cuantifica las respuestas incorrectas del modelo y suele utilizarse para su entrenamiento.

Tienes que entrenar el modelo utilizando la minimización de la función de pérdida. ¿Qué vas a hacer?
Elegir los pesos de regresión lineal de manera que las predicciones se acerquen lo más posible a las respuestas correctas

aᵢ debe ser lo más alto posible para una observación de clase positiva y lo más bajo posible para una observación de clase negativa.

#---------------------------- Gradiente de una función ----------------------------
La función de pérdida depende de los parámetros del algoritmo. 
Esta función es una función de valor vectorial, es decir, toma un vector y devuelve un escalar.

El gradiente de una función de valor vectorial es un vector que está formado por las derivadas de la respuesta para cada argumento
El gradiente de una función para un argumento es la derivada

El gradiente indica la dirección en la que la función crece más rápido. 
Sin embargo, no sirve para resolver el problema de minimización. 
Necesitamos el gradiente negativo, que es un vector opuesto que muestra el decrecimiento más rápido

Para la función unidimensional f(x) = (x - 5)²
el gradiente negativo es ∇f(x)=-2(x-5).
Los valores del gradiente negativo situados a la izquierda de x = 5 son positivos. 
A la derecha son negativos.

El gradiente negativo indica la dirección en la que la función decrece más rápido.
El gradiente de la función en un punto determinado es igual al vector de las derivadas parciales de los argumentos.
El vector gradiente negativo es opuesto al vector gradiente.
El gradiente se convierte en negativo cuando se multiplica por -1.

#---------------------------- Descenso de gradiente ----------------------------
El descenso de gradiente es un algoritmo iterativo para encontrar el mínimo de la función de pérdida. 
Sigue la dirección del gradiente negativo y se aproxima gradualmente al mínimo.
la inmersión debe ser lenta y realizarse con mucho cuidado.
"se sumerge hasta el fondo" al realizar repetidamente los mismos pasos. 
Es difícil llegar al mínimo en una sola iteración porque el vector de gradiente negativo no indica el punto mínimo de la función de pérdida, sino la dirección del decrecimiento.

Para comenzar el descenso, vamos a elegir el valor inicial del argumento (vector x). Se expresa como x⁰.
x¹, se calcula de la siguiente manera: al punto x⁰ se le suma el gradiente negativo multiplicado por el tamaño del paso de descenso del gradiente (μ).

El valor μ determina el tamaño del paso de descenso de gradiente. Si el paso es pequeño, el descenso pasará por muchas iteraciones. No obstante, cada una de ellas nos acercará al mínimo de la función de pérdida. Si el paso es demasiado grande, podemos pasar por alto el mínimo
 El número de iteraciones se expresa como t.

El vector del argumento se actualiza después de cada iteración.
Una vez completado el descenso de gradiente, obtendrás un conjunto de argumentos donde la función de pérdida se acerque al mínimo.

Piensa en esto como un videojuego:
Imagina que estás en una montaña y quieres bajar al valle (encontrar el punto más bajo).

El gradiente es como una brújula que te dice:

¿Hacia dónde caminar? (dirección)
¿Qué tan empinado está? (qué tan rápido cambiar)    

import numpy as np


def func(x):
    return (x[0] + x[1] - 1) ** 2 + (x[0] - x[1] - 2) ** 2


def gradient(x):
    return np.array([4 * x[0] - 6, 4 * x[1] + 2])


print(gradient(np.array([0, 0])))
print(gradient(np.array([0.5, 0.3])))

El gradiente DEBE corresponder exactamente a esa función(a func)
Si la función fuera diferente, el gradiente sería diferente
el gradiente y sus valores dependen completamente de la función func

Cómo el gradiente depende de la función
El gradiente se calcula tomando las derivadas parciales de la función original. En tu caso:

Función original:

func(x) = (x[0] + x[1] - 1)² + (x[0] - x[1] - 2)²
Para obtener el gradiente, calculamos:

Derivada parcial respecto a x[0]: ∂f/∂x₀
Derivada parcial respecto a x[1]: ∂f/∂x₁

import numpy as np


def func(x):
    return (x[0] + x[1] - 1) ** 2 + (x[0] - x[1] - 2) ** 2


def gradient(x):
    return np.array([4 * x[0] - 6, 4 * x[1] + 2])


def gradient_descent(initialization, step_size, iterations):
    x = initialization
    for i in range(iterations):
        x = x - step_size * gradient(x)
    return x


print(gradient_descent(np.array([0, 0]), 0.1, 5))
print(gradient_descent(np.array([0, 0]), 0.1, 100))

#---------------------------- Descenso de gradiente para regresión lineal ----------------------------
y es el vector de respuesta correcta 
a es el vector de predicción.

#---------------------------- Descenso de gradiente estocástico (escenso de gradiente estocástico en minilotes)----------------------------
Podemos calcular el gradiente usando pequeñas partes del conjunto de entrenamiento(minilotes o lotes).
Cada iteracion utiliza un lote diferente de forma aleatoria, asi puede ver todos los datos.

Para obtener lotes, necesitamos barajar todos los datos del conjunto de entrenamiento y dividirlo en partes.
Un lote debe contener un promedio de 128 observaciones (u otro valor potencia de 2)

Cuando el algoritmo DGE ha pasado por todos los lotes una vez, significa que una época ha terminado.
El número de épocas depende del tamaño del conjunto de entrenamiento. 

El número de lotes es igual al número de iteraciones para completar una época.

1. Hiperparámetros de entrada: tamaño del lote, número de épocas y tamaño del paso.
2. Define los valores iniciales de los pesos del modelo.
3. Divide el conjunto de entrenamiento en lotes para cada época.
4. Para cada lote: 
    4.1. Calcula el gradiente de la función de pérdida;
    4.2. Actualiza los pesos del modelo (agrega el gradiente negativo multiplicado por el tamaño del paso a los pesos actuales).
5. El algoritmo devuelve los pesos finales del modelo.

Una época es el conjunto de datos de entrenamiento EN SU TOTALIDAD
1. Divides todo tu conjunto de entrenamiento en pequeños lotes (minilotes)
2. Procesas cada lote uno por uno
3. Cuando terminas todos los lotes = 1 época completa

1000 observaciones en total
Lotes de 128 observaciones cada uno
Necesitas 8 lotes para cubrir todo (1000 ÷ 128 ≈ 8)
Una época = procesar los 8 lotes = ver las 1000 

#---------------------------- Descenso de gradiente estocástico en python ----------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop(['target'], axis=1)
target_train = data_train['target']

data_test = pd.read_csv('/datasets/test_data_n.csv')
features_test = data_test.drop(['target'], axis=1)
target_test = data_test['target']


class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size):
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, train_features, train_target):
        X = np.concatenate(
            (np.ones((train_features.shape[0], 1)), train_features), axis=1
        )
        y = train_target
        w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            batches_count = X.shape[0] // self.batch_size
            for i in range(batches_count):
                begin = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]

                gradient = (
                    2
                    * X_batch.T.dot(X_batch.dot(w) - y_batch)
                    / X_batch.shape[0]
                )

                w -= self.step_size * gradient
        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0


# ya se han superado los parámetros de entrenamiento adecuados
model = SGDLinearRegression(0.01, 1, 200)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

#---------------------------- Regularizacion de regresion lineal ----------------------------
modificar la función de pérdida para eliminar el sobreajuste
"refinar" el modelo si los valores de los parámetros complican la operación del algoritmo
Para un modelo de regresión lineal, la regularización implica la limitación de pesos.

Generalmente, cuanto más bajos sean los valores de los pesos, más fácil es entrenar el algoritmo. 
Para averiguar qué tan grandes son los pesos, calculamos la distancia entre el vector de peso y el vector que consta de ceros.
Cuando usamos la distancia euclidiana para la regularización del peso, dicha regresión lineal se denomina regresión de cresta.

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


data_train = pd.read_csv('/datasets/train_data_n.csv')
features_train = data_train.drop(['target'], axis=1)
target_train = data_train['target']

data_test = pd.read_csv('/datasets/test_data_n.csv')
features_test = data_test.drop(['target'], axis=1)
target_test = data_test['target']


class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size, reg_weight):
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight

    def fit(self, train_features, train_target):
        X = np.concatenate(
            (np.ones((train_features.shape[0], 1)), train_features), axis=1
        )
        y = train_target
        w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            batches_count = X.shape[0] // self.batch_size
            for i in range(batches_count):
                begin = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]

                gradient = (
                    2
                    * X_batch.T.dot(X_batch.dot(w) - y_batch)
                    / X_batch.shape[0]
                )
                reg = 2 * w.copy()
                reg[0] = 0
                gradient += self.reg_weight * reg

                w -= self.step_size * gradient

        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0


print('Regularization:', 0.0)
model = SGDLinearRegression(0.01, 10, 100, 0.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print('Regularization:', 0.1)
model = SGDLinearRegression(0.01, 10, 100, 0.1)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print('Regularization:', 1.0)
model = SGDLinearRegression(0.01, 10, 100, 1.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

print('Regularization:', 10.0)
model = SGDLinearRegression(0.01, 10, 100, 10.0)
model.fit(features_train, target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)
print(r2_score(target_train, pred_train).round(5))
print(r2_score(target_test, pred_test).round(5))

#---------------------------- Fundamentos de redes neuronales ----------------------------
Una red neuronal es un modelo que consta de muchos modelos simples
las neuronas construyen relaciones complejas entre los datos de entrada y salida.

El valor de cada salida, o neurona, se calcula de la misma manera que una predicción de regresión lineal. Aquí, x es un vector, x = (x1, x2, x3)
Cada valor de salida tiene sus propios pesos (w₁ y w₂).

La función logística en una red neuronal se denomina función de activación. Se incluye en la neurona después de multiplicar los valores de entrada por los pesos, cuando las salidas de la neurona se convierten en entradas para otras neuronas. 

Cada variable oculta (h₁, h₂) es igual al valor de entrada multiplicado por un peso

Cuando una red neuronal de este tipo calcula una predicción, realiza todas las operaciones de forma secuencial
Es imposible calcular una predicción de una red neuronal sin valores conocidos de pesos.

La regresión lineal es una red neuronal(solo una neurona)

#---------------------------- Entrenamiento de redes neuronales ----------------------------
necesitamos establecer el objetivo de entrenamiento.

Cualquier red neuronal se puede escribir como una función a partir de su vector de entrada y sus parámetros.

X — características del conjunto de entrenamiento
P — conjunto de todos los parámetros de la red neuronal
N(X, P) — función de red neuronal

Los parámetros de la red neuronal son pesos en las neuronas

y — respuestas del conjunto de entrenamiento
L(a, y) — función de pérdida (por ejemplo, ECM)

El algoritmo de aprendizaje de la red neuronal es el mismo que el algoritmo DGE para la regresión lineal. Solo calculamos el gradiente de la red neuronal, en lugar del gradiente para la regresión lineal.















#---------------------------- Ensambles y potenciación ----------------------------
Que es un ensamblaje?
Un ensamblaje es un conjunto de modelos de aprendizaje automático independientes que trabajan juntos para resolver un problema en particular(todos juntos).
Fortaleza:
    el error medio de un grupo de modelos es menos significativo que sus errores individuales
Tipos de modelos de ensamble: 
    bosque aleatorio (se promedian los resultados de los alumnos base o débiles (modelos que componen el ensamble))
        En un bosque aleatorio, los alumnos base son los árboles de decisión individuales que forman el bosque.
        Se llaman "débiles" porque:
            Cada árbol individual no es muy bueno por sí solo
            Pueden cometer errores o tener predicciones imprecisas
            Son "simples" comparados con el modelo completo
        Aunque cada árbol individual sea "débil", cuando combinas muchos árboles débiles, el resultado es mucho más fuerte y preciso.
    potenciación
        cada modelo posterior considera los errores del anterior y, en la predicción final, los pronósticos de los alumnos básicos
¿Cómo funciona paso a paso?
Paso 1: Primer modelo

Entrenas el primer modelo (alumno base b₁)
Hace predicciones, pero comete errores
Calculas los residuos = diferencia entre predicción y respuesta real
Paso 2: Segundo modelo

El segundo modelo (b₂) NO intenta predecir el objetivo original
En su lugar, intenta predecir los errores del primer modelo
Se entrena específicamente para corregir lo que el primero hizo mal
Paso 3: Tercer modelo y siguientes

Cada nuevo modelo se enfoca en corregir los errores del ensamble anterior
Es como tener un equipo de correctores trabajando en secuencia

La predicción final es la suma de todos los modelos:

Predicción final = b₁(x) + b₂(x) + b₃(x) + ... + bₙ(x)

Bosque aleatorio: Modelos independientes → se promedian
Potenciación: Modelos secuenciales → cada uno corrige al anterior → se suman

En cada paso siguiente, el algoritmo minimiza el error de ensamble del paso anterior.
El ensamble en sí se representa como la suma de las predicciones de todos los alumnos base combinados

¿Qué son los residuos del ensamble en el paso N?

La diferencia entre el ensamble en el paso actual N y las respuestas reales


#---------------------------- Potenciación del gradiente ----------------------------
Si combinamos la potenciación y el descenso de gradiente

En la potenciación, cada alumno base intenta "empujar" las predicciones del paso anterior hacia las respuestas correctas. Así es como se minimiza la función de pérdida.
ahora las respuestas del modelo son el argumento a lo largo del cual se realizará el descenso

En cada paso, selecciona las respuestas que minimizarán la función
Minimiza la función con descenso de gradiente. Para ello, en cada paso, calcula el gradiente negativo de la función de pérdida para la predicción gN
Para impulsar las predicciones hacia las respuestas correctas, el alumno base aprende a predecir gN
Obtén el peso para bN de la tarea de minimización iterando diferentes valores
Es el coeficiente para el alumno base lo que ayuda a ajustar el ensamble para hacer predicciones lo más precisas posible.

#---------------------------- Regularización de potenciación del gradiente ----------------------------
reducir el sobreajuste durante la potenciación del gradiente.

reducción del tamaño del paso;
ajuste de los parámetros del árbol;
aleatorización de submuestras para estudiantes base bᵢ.

Cuando un algoritmo toma pasos que son demasiado grandes, este recuerda rápidamente el conjunto de entrenamiento. Esto da como resultado un sobreajuste del modelo.

Introduce el coeficiente η, que controla la tasa de aprendizaje y puede usarse para reducir el tamaño del paso


El valor de este coeficiente se elige iterando diferentes valores en el rango de 0 a 1. Un valor más pequeño significa un paso más pequeño hacia el gradiente negativo y una mayor exactitud del ensamble. Pero si la tasa de aprendizaje es demasiado baja, el proceso de entrenamiento llevará demasiado tiempo.


La segunda forma de regularizar la potenciación del gradiente es ajustar los parámetros del árbol. Podemos limitar la profundidad del árbol o la cantidad de elementos en cada nodo, probar diferentes valores y ver cómo afecta el resultado. 
El tercer método de regularización es trabajar con submuestras. El algoritmo funciona con submuestras en lugar del conjunto completo. Esta versión del algoritmo es similar al DGE y se llama potenciación del gradiente estocástico.

#---------------------------- Lbrerias para potenciación del gradiente ----------------------------
CatBoost funciona con características categóricas
Librería	Tratamiento de características categóricas	Velocidad
XGBoost	No	Baja
LightGBM	Sí	Alta
CatBoost	Sí	Alta

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


data = pd.read_csv('/datasets/travel_insurance_us.csv')

features_train, features_valid, target_train, target_valid = train_test_split(
    data.drop('Claim', axis=1), data.Claim, test_size=0.25, random_state=12345
)

cat_features = [
    'Agency',
    'Agency Type',
    'Distribution Channel',
    'Product Name',
    'Destination',
    'Gender',
]

model = CatBoostClassifier(loss_function="Logloss", iterations=150, random_seed=12345)

model.fit(features_train, target_train, cat_features=cat_features, verbose=10)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
print(roc_auc_score(target_valid, probabilities_one_valid))

CatBoostClassifier NO es exclusivamente para características categóricas.
CatBoost es un algoritmo de potenciación (boosting) que puede trabajar con:

Características numéricas ✅
Características categóricas ✅
Ambas al mismo tiempo
es especialmente bueno manejando características categóricas, pero no se limita solo a ellas.

Manejo automático de características categóricas sin necesidad de:

One-hot encoding manual
Label encoding manual
Otras transformaciones complicadas

#---------------------------- Trabajar con solicitudes  ----------------------------
1. Asegúrate de saber quién es tu cliente final
2. Averigua cuál es el objetivo final de tu trabajo.
3. Especifica todos los detalles que el cliente de la tarea desea conocer. ¿Hay inconvenientes ocultos?
4. Repasa todos los detalles y escribe cuál debería ser el resultado perfecto
    te recomendamos anotar los parámetros de entrada para que tengas algo que consultar.
5. Esbozar un prototipo y consultar con el cliente para ver si este funciona para ellos.
    Puede haber varias iteraciones de "prototipo —> retroalimentación —> correcciones" antes de acordar un formulario conveniente.
6. Proporcionar al cliente el resultado final.



































































































































































































































#---------------------------- Complejidad computacional ----------------------------
















































































































































































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------
















































#XGBOOST - modelo final
def modelo_XGBOOSTr(features_train, target_train):
    target_train_np = target_train.values.flatten()
    
    best_smape = float("inf") #representa el infinito positivo, el numero mas grande que cualquier otro
    best_est = None
    best_depth = None
    best_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=12345)

    for est in range(130, 150, 20):
        for depth in range(15, 21, 2):

            smape_folds = []

            for train_idx, val_idx in kf.split(features_train):
                
                features_train_cv = features_train.iloc[train_idx]
                features_val_cv   = features_train.iloc[val_idx]
            
                target_train_cv = target_train_np[train_idx]
                target_val_cv   = target_train_np[val_idx]

                xgb_model = XGBRegressor(
                    n_estimators=est,
                    learning_rate=0.05,
                    max_depth=depth,
                    subsample=0.8, #porcentaje de las filas del dataset se usa para entrenar cada árbol, elegidas aleatoriamente
                    colsample_bytree=0.8, #porcentaje de columnas (features) se usa para construir cada árbol, al azar
                    random_state=12345,
                    n_jobs=-1
                )
                
                xgb_model.fit(features_train_cv, target_train_cv)
                target_pred_cv = xgb_model.predict(features_val_cv)

                smape_cv = calcular_smape(target_val_cv, target_pred_cv)
                smape_folds.append(smape_cv)

            smape_mean = np.mean(smape_folds)

            print(f"n_estimators={est} | max_depth={depth} | sMAPE CV={smape_mean:.4f}")

            if smape_mean < best_smape:
                best_smape = smape_mean
                best_est = est
                best_depth = depth

  
    best_model = XGBRegressor(
        n_estimators=best_est,
        learning_rate=0.05,
        max_depth=best_depth,
        subsample=0.8, #porcentaje de las filas del dataset se usa para entrenar cada árbol, elegidas aleatoriamente
        colsample_bytree=0.8, #porcentaje de columnas (features) se usa para construir cada árbol, al azar
        random_state=12345,
        n_jobs=-1
    )

    best_model.fit(features_train, target_train_np) #ahora sin las particiones y con los mejores parametros

    print()
    print("mejor configuracion (CV)")
    print(f"sMAPE CV promedio: {best_smape:.4f}")
    print(f"n_estimators: {best_est}")
    print(f"max_depth: {best_depth}")

    return best_model, best_smape
























































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------








































































































































#---------------------------- Complejidad computacional ----------------------------









































































































































