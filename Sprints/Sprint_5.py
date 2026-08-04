#------------Importacion de libreria----------------
import seaborn as sns
from scipy import stats as st
import math as mt
from math import factorial
import numpy as np
import pandas as pd

#------------Diagrama de caja----------------
# Resumir la distribución de una variable numérica a través de sus cuartiles

dataset = pd.Series([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 6])
sns.boxplot(dataset)

# Diferencia entre varianza y covarianza
# La varianza mide la dispersión de una sola variable respecto a su propia media
# La covarianza mide cómo cambian o varían dos variables distintas en conjunto.

#------------Varianza----------------
data = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
variance = np.var(data)
print(variance)

#------------Covarianza----------------
x = [1, 2, 3, 4, 5, 6] # dataset 1
y = [41, 62, 89, 96, 108, 115] # dataset 2
covariance_matrix = np.cov(x,y) # calculamos la matriz de covarianza
covariance = covariance_matrix[0][1] # extraemos la covarianza como valor
print(covariance)

#------------Desviacion estandar----------------
s = pd.Series([1, 2, 3, 4, 5, 6])
s.describe()

x = [1, 2, 3, 4, 5, 6]
standard_deviation = np.std(x)
print(standard_deviation)

# La diferencia fundamental entre la varianza y la desviación estándar radica en la unidad de medida y la interpretabilidad.
# Ambas miden la dispersión o variabilidad de un conjunto de datos respecto a su media, pero expresan esa variación en escalas distintas.

# Varianza: Promedio de las desviaciones individuales al cuadrado respecto a la media.
# Desviación Estándar: Raíz cuadrada de la varianza.

#------------Distribucion normal  o de gauss, regla de las 3 sigmas----------------

# Los datos situados cerca de la media aparecen con más frecuencia
# La desviación estándar, que determina la dispersión con respecto a la media.
# Una desviación estándar menor hace que el gráfico sea más empinado
# (μ−3σ,μ+3σ); µ es la media y σ es la desviación estándar; limite inferior y superior

# Cálculo de un umbral máximo de tiempo (adv_time)
    # calcular un límite superior esperable
adv_mean = 3
adv_var = 0.25
adv_std = np.sqrt(adv_var)
adv_time = adv_mean + 3 * adv_std

# Cálculo de probabilidades de visitantes (st.norm.cdf)
    # Analiza la probabilidad de tener eventos extremos, limite superior e inferior
mu = 100500
sigma = 3500
more_threshold = 111000
fewer_threshold = 92000
p_more_visitors = 1 - st.norm(mu, sigma).cdf(more_threshold)
p_fewer_visitors = st.norm(mu, sigma).cdf(fewer_threshold)
print(f'Probabilidad de que el número de visitantes sea superior a {more_threshold}: {p_more_visitors}')
print(f'Probabilidad de que el número de visitantes sea inferior a {fewer_threshold}: {p_fewer_visitors}')

mu = 420
sigma = 65
prob = 0.9
n_shipment = st.norm(mu, sigma).ppf(prob)
print('Cantidad de artículos a pedir:', int(n_shipment))

mu = 24
sigma = 3.2
threshold = .75
max_delivery_price = st.norm(mu, sigma).ppf(1 - threshold)
print('Costo máximo de envío por mensajería:', max_delivery_price)

quiz_mean = 6000 / 100 * 3
quiz_std = 6000 / 100 * 0.4
quiz_bottom_line = quiz_mean - 3 * quiz_std
quiz_top_line = quiz_mean + 3 * quiz_std
print('Intervalo:', quiz_bottom_line, '-', quiz_top_line) 


# Establecemos los valores de los parámetros
binom_n = 5000
binom_p = 0.15
clicks = 715

# calculamos el valor esperado de éxitos y sigma
# (Mu) — La Media poblacional o el promedio aritmético
# (Sigma) — La Desviación Estándar poblacional

mu = binom_n * binom_p
sigma = mt.sqrt(binom_n * binom_p * (1 - binom_p))

# calculamos la probabilidad de obtener 715 visitas o menos
p_clicks = st.norm(mu, sigma).cdf(clicks)
print(p_clicks)

# Una distribución de datos en la que la mayor parte de los valores se encuentran a la derecha se denomina sesgo a la derecha(asimetría positiva.)

#------------Valor atipico----------------
# si está a más de 1.5 veces el rango intercuartílico (IQR) de alguno de los cuartiles Q1 o Q3

#--------------Probabilidad de evento--------------
# P para denotar probabilidad y la letra A para denotar un evento 
# P(A) = número de resultados favorables / número total de resultados posibles
# espacio muestral = conjunto de todos los resultados posibles de un experimento, denotado con S.

print(
    len(cool_rock[cool_rock['Song'] == 'Smells Like Teen Spirit']) # Cuantas veces esta sobre cuantos hay en total
    / len(cool_rock)
)

#------------Distribuccion normal----------------
# Tiene dos parámetros clave: la media y la varianza
# Los datos se distribuyen simétricamente alrededor de la media

#------------La función de distribución acumulativa----------------

# Da la probabilidad de que una variable aleatoria sea menor o igual que un valor determinado
# ¿Cuál es la probabilidad de que un estudiante pueda aprender una nueva profesión por menos de 4 000 dólares?

# norm.cdf()

data = st.norm(5000, 1500) # media de 5 000 y una desviación estándar de 1500.
desired_cost = 4000 # el coste deseado.
probability = data.cdf(desired_cost) # calculamos la probabilidad de obtener el valor desired_cost.


binom_n = 23000 # Total de pruebas
binom_p = 0.4 # Probabilidad de que ocurra

threshold = 9000 # Media de que ocurrra

mu = binom_n * binom_p #media
sigma = mt.sqrt(binom_n * binom_p * (1 - binom_p)) # Desviacion estandar

p_threshold = 1 - st.norm(mu, sigma).cdf(threshold) #1 es el total 
print(p_threshold)

#------------ La función de punto porcentual----------------
# Da el valor de la variable aleatoria que corresponde a una determinada probabilidad	
# ¿Cuál es el coste máximo de la formación para el 10% de los estudiantes que gastaron menos dinero en sus estudios?

# norm.ppf()	

data = st.norm(5000, 1500)
target_level = 0.1 # 10% de los estudiantes que gastaron menos dinero.
cost = data.ppf(target_level) # Encontramos el importe que no supere los gastos del 10% de estudiantes que gastaron menos dinero.


# La puntuación media del examen es 1000 y la desviación estándar es 100. 
# Hay que encontrar la probabilidad de obtener entre 900 y 1 100 puntos en el examen.

st.norm(1000, 100).cdf(1100) - st.norm(1000, 100).cdf(900)

# calcular la probabilidad de que el valor que buscamos sea inferior a 1 100 y restarle la probabilidad de que sea inferior a 900

#------------PERMUTACIONES----------------

#  sirve para ordenar objetos, donde el orden de las cosas sí importa y no se repiten
# Cada combinacion es nueva y diferente, sin repetir

# número de permutaciones de n objetos es: n! ; n!=1⋅2⋅3⋅...⋅(n−1)⋅n
courses_amount = 5
result = factorial(courses_amount)

#------------Combinaciones----------------
# resultado = n!/k!(n−k)!; 10!/3!⋅(10−3)!
n = 10
k = 3
combinations = factorial(n) / (factorial(k) * factorial(n-k))

#------------ Distribucion binomial (calcular numero de exito en numero de pruebas)----------------

# calcular la probabilidad de obtener un número determinado de éxitos en un número fijo de pruebas
    # número fijo de ensayos
    # los resultados binarios
    # la independencia entre pruebas

# P=p**k * q **(n-k)
# p es la probabilidad de éxito. q es la probabilidad de fracaso. k es el número de pruebas realizadas con éxito. n es el número total de pruebas.
p = 0.6
q = 0.4
n = 80
k = 50
probability = factorial(n) / (factorial(k) * factorial(n-k)) * (p ** k) * (q ** (n-k))
print(probability)

#------------Array en numpy----------------
# ndarray = Inmutable una vez creado
data = np.array([1, 3, 5, 7, 11, 13, 17, 19, 23, 317])
print('El primer elemento:', data[0])
print('El último elemento:', data[-1])

# Arreglo con 20 números aleatorios
data = np.random.normal(size = 20) 
print(data)

# . The numbers follow a normal distribution, centered around a mean of 15 with a standard deviation of 5.
mean = 15
std_dev = 5
data = np.random.normal(mean, std_dev, size = 20) 
print(data)

#------------Agrupando segun valores----------------

failed_students = 0
for score in exam_results:
    if score < 20:
        failed_students += 1
print('Número de estudiantes reprobados:', failed_students)

exam_results = np.array(
    [
        42,  56,  59,  76,  43,  34,  62,  51,  50,  65,  
        66,  50,  46,  5,  79, 99,  51,  26,  35,   8,  
        34,  47,  64,  58,  61,  12,  30,  63,  20,  68
    ]
)

summarized_data = {'excellent': 0, 'good': 0, 'average': 0, 'passable': 0, 'failed': 0}

for score in exam_results:
    if score >= 90:
        summarized_data['excellent'] += 1
    elif score >= 70:
        summarized_data['good'] += 1
    elif score >= 50:
        summarized_data['average'] += 1
    elif score >= 20:
        summarized_data['passable'] += 1
    else:
        summarized_data['failed'] += 1

for result in summarized_data:
    print(result, '-', summarized_data[result])   

#------------Muestras aleatorias y medias muestrales---------------

# Media muestral de la distribución muestral: Conjunto de valores medios de todas las muestras posibles de un cierto tamaño tomadas de una población estadística particular

# Cuanto mayor sea, menor será la desviación estándar de la media de la muestra
# Cuanto mayor sea la muestra que tomemos, más precisa será la media

# Error estándar (s/root(n)); s desviacion estandar, n tamaño de la muestra: desviacion estandar de la media, respecto a la muestra promedio poblacional

#------------Formular hipótesis de dos colas---------------
# En estadística, H₀ suele expresar que no hay diferencias entre grupos. 
# Esta hipótesis nula supone que no hay ningún cambio hasta que se demuestre lo contrario.

# La hipótesis alternativa, H₁, contradice la hipótesis nula.
# Afirma que hay una diferencia entre los grupos

#------------Prueba de hipótesis en Python. Valores p---------------
# Estadística de diferencia entre la media y el valor con el que lo estás comparando

# Si este valor es superior al 10%, definitivamente no debes rechazar la hipótesis nula. 
# Si el valor p es más bajo, es posible que debas rechazar la hipótesis nula. 

# Los valores de umbral convencionales son 5% y 1%.

# Leer datos
time_on_site = pd.read_csv('user_time.csv')

interested_value = 120 # tiempo transcurrido en el sitio web
alpha = .05 # umbral

results = st.ttest_1samp(
    time_on_site, 
    interested_value
)

print('valor p: ', results.pvalue) # Imprimir el valor p resultante
if (results.pvalue < alpha): # Comprobacion
    print("Rechazamos la hipótesis nula") # resultado diferente de interested_value
else:
    print("No podemos rechazar la hipótesis nula") # el interested_value tiene razon

#Si la probabilidad es relativamente alta, los datos no nos dan motivos para rechazar una suposición.
#Si la probabilidad es baja, entonces a partir de los datos proporcionados podemos concluir que nuestra suposición probablemente fue incorrecta (pero no podemos rechazarla o probar lo contrario).

#------------Formular hipótesis de una cola---------------
# python por defecto hace hipotesis de dos colas, 
# para una unilateral: divide el valor p entre dos

screens = pd.Series([
    4, 2, 4, 5, 5, 4, 2, 3, 3, 5, 2, 5, 2, 2, 2, 3, 3, 4, 8, 3, 4, 3, 5, 5, 4, 2, 5, 2, 
    3, 7, 5, 5, 6,  5, 3, 4, 3, 6, 3, 4, 4, 3, 5, 4, 4, 8, 4, 7, 4, 5, 5, 3, 4, 6, 7, 2,
    3, 6, 5, 6, 4, 4, 3, 4, 6, 4, 4, 6, 2, 6, 5, 3, 3, 3, 4, 5, 3, 5, 5, 4, 3, 3, 3, 1, 
    5, 4, 3, 4, 6, 3, 1, 3, 2, 7, 3, 6, 6, 6, 5, 5
    ])

prev_screens_value = 4.867 # promedio contra el cual hacer la hipotesis
alpha = 0.05  # nivel de significación
results = st.ttest_1samp(screens, prev_screens_value)

print('valor-p: ', results.pvalue / 2) 

# prueba unilateral a la izquiera:

# rechaza la hipótesis solo si la media muestral es significativamente menor que el valor propuesto
if (results.pvalue / 2 < alpha) and (screens.mean() < prev_screens_value): # Comparamos el valor p con el umbral y la media muestral con el valor propuesto
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

#Si la prueba fuera unilateral a la derecha con la hipótesis alternativa:
# “El valor observado es mayor que el predicho”, uno de los signos “<” cambiaría a “>”. 
# Las últimas líneas de código se verían así:

if (results.pvalue / 2 < alpha) and (screens.mean() > prev_screens_value):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


revenue = pd.Series([727, 678, 685, 669, 661, 705, 701, 717, 
                     655,643, 660, 709, 701, 681, 716, 655, 
                     716, 695, 684, 687, 669,647, 721, 681, 
                     674, 641, 704, 717, 656, 725, 684, 665])

interested_value = 800
alpha = 0.05
results = st.ttest_1samp(revenue, interested_value)

print('valor p:', results.pvalue / 2)

if (results.pvalue / 2 < alpha) and (revenue.mean() < interested_value):
    print(
        "Rechazamos la hipótesis nula: los ingresos fueron significativamente inferiores a 800 dólares"
)
else:
    print(
        "No podemos rechazar la hipótesis nula: los ingresos no fueron significativamente inferiores"
)

#------------Hipótesis sobre la igualdad de las medias de dos poblaciones---------------

# Dos muestras independientes

# equal_var: Parámetro opcional que especifica si las varianzas de las poblaciones deben considerarse iguales (True o False)
# True y la varianza de cada muestra se estimará a partir del dataset combinado de las dos muestras

sample_1 = [3071, 3636, 3454, 3151, 2185, 3259, 1727, 2263, 2015, 
            2582, 4815, 633, 3186, 887, 2028, 3589, 2564, 1422, 1785, 
            3180, 1770, 2716, 2546, 1848, 4644, 3134, 475, 2686, 
            1838, 3352]
sample_2 = [1211, 1228, 2157, 3699, 600, 1898, 1688, 1420, 5048, 3007, 
            509, 3777, 5583, 3949, 121, 1674, 4300, 1338, 3066, 
            3562, 1010, 2311, 462, 863, 2021, 528, 1849, 255, 
            1740, 2596]

alpha = 0.05  
results = st.ttest_ind(sample_1, sample_2) # si el valor p es menor que alpha, rechazamos la hipótesis

print('valor p: ', results.pvalue) 
if results.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# tiempo pasado en el sitio web por usuarios con un nombre de usuario y contraseña
time_on_site_logpass = [368, 113, 328, 447, 1, 156, 335, 233, 
                       308, 181, 271, 239, 411, 293, 303, 
                       206, 196, 203, 311, 205, 297, 529, 
                       373, 217, 416, 206, 1, 128, 16, 214]
# tiempo pasado en el sitio web por los usuarios que inician sesión a través de las redes sociales
time_on_site_social  = [451, 182, 469, 546, 396, 630, 206, 
                        130, 45, 569, 434, 321, 374, 149, 
                        721, 350, 347, 446, 406, 365, 203, 
                        405, 631, 545, 584, 248, 171, 309, 
                        338, 505]

alpha = .05
results = st.ttest_ind(time_on_site_logpass, time_on_site_social)

print('valor p:', results.pvalue)
if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


pages_per_session_autumn = [7.1, 7.3, 9.8, 7.3, 6.4, 10.5, 8.7, 
                            17.5, 3.3, 15.5, 16.2, 0.4, 8.3, 
                            8.1, 3.0, 6.1, 4.4, 18.8, 14.7, 16.4, 
                            13.6, 4.4, 7.4, 12.4, 3.9, 13.6, 
                            8.8, 8.1, 13.6, 12.2]
pages_per_session_summer = [12.1, 24.3, 6.4, 19.9, 19.7, 12.5, 17.6, 
                            5.0, 22.4, 13.5, 10.8, 23.4, 9.4, 3.7, 
                            2.5, 19.8, 4.8, 29.0, 1.7, 28.6, 16.7, 
                            14.2, 10.6, 18.2, 14.7, 23.8, 15.9, 16.2, 
                            12.1, 14.5]

alpha = .05
results = st.ttest_ind(pages_per_session_autumn, pages_per_session_summer, equal_var=False)

print('valor p:', results.pvalue)
if (results.pvalue < alpha):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

#-----------Que es la varianza?---------------

# La varianza mide qué tan dispersos o separados están los datos de su promedio. 

# Grupo A: 8, 8, 8, 8, 8 (promedio = 8) Varianza baja (todos muy similares)
# Grupo B: 2, 6, 8, 12, 12 (promedio = 8) Varianza alta (muy dispersos)

# Puedes calcularla así:
varianza_logpass = np.var(time_on_site_logpass)
varianza_social = np.var(time_on_site_social)

#-----------Hipótesis sobre la igualdad de las medias de muestras emparejadas---------------

# Dos muestras relacionadas/pareadas
# Significa que medimos una variable dos veces para cada cliente, antes y después de los cambios.
# Deben tener el mismo tamaño

before = [157, 114, 152, 355, 155, 513, 299, 268, 164, 320, 
                    192, 262, 506, 240, 364, 179, 246, 427, 187, 431, 
                    320, 193, 313, 347, 312, 92, 177, 225, 242, 312]
after = [282, 220, 162, 226, 296, 479, 248, 322, 298, 418, 
                 552, 246, 251, 404, 368, 484, 358, 264, 359, 410, 
                 382, 350, 406, 416, 438, 364, 283, 314, 420, 218]

alpha = 0.05  
results = st.ttest_rel(before, after)
print('valor p: ', results.pvalue)

if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula") # ha habido un cambio 
else:
    print("No podemos rechazar la hipótesis nula") # no ha habido un cambio


# Seleccionamos una prueba unilateral debido a la palabra "más". 
# Si no sabemos la dirección del cambio, entonces utilizaremos la prueba bilateral.

bullets_before = [821, 1164, 598, 854, 455, 1220, 161, 1400, 479, 215, 
          564, 159, 920, 173, 276, 444, 273, 711, 291, 880, 
          892, 712, 16, 476, 498, 9, 1251, 938, 389, 513]
bullets_after = [904, 220, 676, 459, 299, 659, 1698, 1120, 514, 1086, 1499, 
         1262, 829, 476, 1149, 996, 1247, 1117, 1324, 532, 1458, 898, 
         1837, 455, 1667, 898, 474, 558, 639, 1012]

print('media anterior:', pd.Series(bullets_before).mean())
print('media posterior:', pd.Series(bullets_after).mean())

alpha = 0.05
results = st.ttest_rel(
    bullets_before, 
    bullets_after)
print('valor-p:', results.pvalue/2)

if results.pvalue/2 < alpha:
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

#-----------Muestreo estratificado---------------
# una técnica de muestreo donde dividimos la población en grupos más pequeños llamados estratos (que comparten características similares)
# luego tomamos muestras aleatorias de cada estrato.

#-----------Teorema del límite central--------------
# Si tomas muchas muestras de cualquier población y calculas la media de cada muestra, esas medias se distribuirán de forma normal (curva de campana)
# sin importar cómo se vea la población original.

#-----------¿Qué es el nivel de significancia?--------------
# es el umbral que establecemos antes de realizar una prueba estadística para decidir si rechazamos o no la hipótesis nula.

#-----------Función de densidad de probabilidad--------------
# una curva que describe todas las posibilidades de que ocurra un evento
# Dónde es más probable que ocurran ciertos valores
# Dónde es menos probable que ocurran otros valores

#-----------¿Qué es una prueba t?--------------
# es un método estadístico que te ayuda a determinar si existe una diferencia significativa
# tiene "colas más gruesas", lo que la hace más apropiada cuando trabajas con muestras pequeñas (menos de 30 observaciones) o cuando no conoces la desviación estándar de toda la población.
# la hipótesis nula siempre implica la ausencia de diferencias.
# Si el valor p es mayor que el valor alfa ⇒ no podemos rechazar H₀.