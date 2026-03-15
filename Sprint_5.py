data.hist(bins=4, alpha=0.5)  # crea un histograma con cuatro contenedores

data.hist(
    bins=[11, 20, 30, 40, 50, 60, 70, 80, 90, 99], alpha=0.7 
    #al usar bins con listas, se definen los contenedores de forma personalizada, donde ira cada uno para visualizar mejor todos los datos
#de 11 a 20, de 20 a 30, etc
)

data = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print('La media es igual a', data.mean())
print('La mediana es igual a', data.median())

#------------Importacion de libreria----------------
import seaborn as sns

#------------Diagrama de caja----------------
dataset = pd.Series([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 6])
sns.boxplot(dataset)

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

x = [1, 2, 3, 4, 5, 6] # dataset
standard_deviation = np.std(x)
print(standard_deviation)

variance = 3.5 #si ya conoces la varianza de un dataset
standard_deviation = np.sqrt(variance)
print(standard_deviation)

#------------Distribucion normal  o de gauss, regla de las 3 sigmas----------------
#los datos situados cerca de la media aparecen con más frecuencia
#La desviación estándar, que determina la dispersión con respecto a la media.
#una desviación estándar menor (en comparación con la media) hace que el gráfico sea más empinado
#(μ−3σ,μ+3σ); µ es la media y σ es la desviación estándar; limite inferior y superior
adv_mean = 3
adv_var = 0.25
adv_std = np.sqrt(adv_var)
adv_time = adv_mean + 3 * adv_std

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

# importamos las librerías necesarias
from scipy import stats as st
import math as mt
# establecemos los valores de los parámetros
binom_n = 5000
binom_p = 0.15
clicks = 715
# calculamos el valor esperado de éxitos y sigma
mu = binom_n * binom_p
sigma = mt.sqrt(binom_n * binom_p * (1 - binom_p))
# calculamos la probabilidad de obtener 715 visitas o menos
p_clicks = st.norm(mu, sigma).cdf(clicks)
print(p_clicks)

#------------FUNCIONES EN PANDAS BASE----------------


#Una distribución de datos en la que la mayor parte de los valores se encuentran a la derecha se denomina sesgo a la derecha(asimetría positiva.)


#------------Valor atipico----------------
#si está a más de 1.5 veces el rango intercuartílico (IQR) de alguno de los cuartiles Q1 o Q3,  IQR es Q3 - Q1



#------------Dataframes sesgados----------------
#los datasets sesgados a la derecha, la media es mayor que la mediana, y viceversa


#--------------Probabilidad de evento--------------
#P para denotar probabilidad y la letra A para denotar un evento 
# P(A) = número de resultados favorables / número total de resultados posibles
#espacio muestral = conjunto de todos los resultados posibles de un experimento, denotado con S.

print(
    len(cool_rock[cool_rock['Song'] == 'Smells Like Teen Spirit']) #cuantas veces esta sobre cuantos hay en total
    / len(cool_rock)
)




P(A) = 1/6 y P(B) = 1/6. Siguiendo la fórmula que te hemos proporcionado, podemos calcular P(A and B) = 1/6 x 1/6 = 1/36.
La probabilidad de que uno de los usuarios aleatorios sea hombre es 0.29. La probabilidad de que el otro usuario aleatorio no haya indicado su género es (1 - 0.29 - 0.25) = 0.46. Estos son eventos independientes. Esto significa que la probabilidad de que ocurran ambos eventos es 0.46 * 0.29 = 0.1334, o 13.34%.
La probabilidad de elegir una de las 4 secciones posibles es 1/4. La probabilidad de elegir una de las 3 subsecciones posibles es 1/3. La probabilidad de elegir una de las 5 páginas web que nos interesan del total de 9 páginas web en la subsección es 5/9. Respuesta: 1/4 * 1/3 * 5/9 = 5/108 ≈ 0.046296 ≈ 4.63%

#------------Distribuccion normal----------------
#tiene dos parámetros clave: la media y la varianza
#los datos se distribuyen simétricamente alrededor de la media


#------------Importar libreria----------------
from scipy import stats as st

#------------La función de distribución acumulativa----------------
#Da la probabilidad de que una variable aleatoria sea menor o igual que un valor determinado
#¿Cuál es la probabilidad de que un estudiante pueda aprender una nueva profesión por menos de 4 000 dólares?
#norm.cdf()

data = st.norm(5000, 1500) # media de 5 000 y una desviación estándar de 1500.
desired_cost = 4000# el coste deseado.
probability = data.cdf(desired_cost)# calculamos la probabilidad de obtener el valor desired_cost.

from scipy import stats as st
import math as mt

binom_n = 23000 #total de pruebas
binom_p = 0.4 #probabilidad de que ocurra

threshold = 9000 #media de que ocurrra

mu = binom_n * binom_p #media
sigma = mt.sqrt(binom_n * binom_p * (1 - binom_p)) #desviacion estandar

p_threshold = 1 - st.norm(mu, sigma).cdf(threshold) #1 es el total 
print(p_threshold)

#------------La función de punto porcentual----------------
#Da el valor de la variable aleatoria que corresponde a una determinada probabilidad	
#¿Cuál es el coste máximo de la formación para el 10% de los estudiantes que gastaron menos dinero en sus estudios?
#norm.ppf()	

data = st.norm(5000, 1500)
target_level = 0.1 # para el 10% de los estudiantes que gastaron menos dinero.
cost = data.ppf(target_level)# encontramos el importe que no supere los gastos del 10% de estudiantes que gastaron menos dinero.

#La puntuación media en el examen del Certificado de análisis de datos es 1000 y la desviación estándar es 100. Hay que encontrar la probabilidad de obtener entre 900 y 1 100 puntos en el examen.
st.norm(1000, 100).cdf(1100) - st.norm(1000, 100).cdf(900)
#calcular la probabilidad de que el valor que buscamos sea inferior a 1 100 y restarle la probabilidad de que sea inferior a 900

#------------PERMUTACIONES----------------
#número de permutaciones de n objetos es n! ; n!=1⋅2⋅3⋅...⋅(n−1)⋅n
courses_amount = 5
result = factorial(courses_amount)

#------------Libreria matematicas----------------
from math import factorial

#------------Combinaciones----------------
#resultado = n!/k!(n−k)!; 10!/3!⋅(10−3)!
n = 10
k = 3
combinations = factorial(n) / (factorial(k) * factorial(n-k))

#------------Distribucion binomial; tambien calcular numero de exito en numero de pruebas----------------
#hay dos resultados posibles; éxito es p, e fracaso = (1-p), pruebas independientes entre si
#P=p**k * q **(n-k)
# p es la probabilidad de éxito. q es la probabilidad de fracaso. k es el número de pruebas realizadas con éxito. n es el número total de pruebas.
p = 0.6
q = 0.4
n = 80
k = 50
probability = factorial(n) / (factorial(k) * factorial(n-k)) * (p ** k) * (q ** (n-k))
print(probability)

#------------Array en numpy----------------
#ndarray = el tamaño se determina en la creación y no puede modificarse durante la ejecución
data = np.array([1, 3, 5, 7, 11, 13, 17, 19, 23, 317])
print('El primer elemento:', data[0])
print('El último elemento:', data[-1])

data = np.random.normal(size = 20) #array de 20 números que sigan una distribución normal
print(data)

mean = 15
std_dev = 5
data = np.random.normal(mean, std_dev, size = 20) #array con una media y una desviación estándar dadas
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
#la distribución muestral de la media muestral es el conjunto de valores medios de todas las muestras posibles de un cierto tamaño tomadas de una población estadística particular
#Cuanto mayor sea, menor será la desviación estándar de la media de la muestra
#Cuanto mayor sea la muestra que tomemos, más precisa será la media

#La desviación estándar de la media de la muestra del promedio de la población estadística se llama error estándar (s/root(n)); s desviacion estandar, n tamaño de la muestra

#------------Formular hipótesis de dos colas---------------
#En estadística, H₀ suele expresar que no hay diferencias entre grupos. Esta hipótesis nula supone que no hay ningún cambio hasta que se demuestre lo contrario
#La hipótesis alternativa, H₁,contradice la hipótesis nula. Afirma que hay una diferencia entre los grupos

#------------Prueba de hipótesis en Python. Valores p---------------
#estadística de diferencia entre la media y el valor con el que lo estás comparando
#Si este valor es superior al 10%, definitivamente no debes rechazar la hipótesis nula. Si el valor p es más bajo, es posible que debas rechazar la hipótesis nula. Los valores de umbral convencionales son 5% y 1%.
from scipy import stats as st
import numpy as np
import pandas as pd
time_on_site = pd.read_csv('user_time.csv')
interested_value = 120 # tiempo transcurrido en el sitio web
alpha = .05 # la significancia estadística crítica (umbral)
results = st.ttest_1samp(# realizar una prueba, esta es la funcion
    time_on_site, 
    interested_value)
print('valor p: ', results.pvalue)# imprimir el valor p resultante
if (results.pvalue < alpha):# comparar el valor p con el umbral
    print("Rechazamos la hipótesis nula") #no se pasan 130 minutos en el sitio web
else:
    print("No podemos rechazar la hipótesis nula") #se pasan 120 minutos en el sitio web

#Si la probabilidad es relativamente alta, los datos no nos dan motivos para rechazar una suposición.
#Si la probabilidad es baja, entonces a partir de los datos proporcionados podemos concluir que nuestra suposición probablemente fue incorrecta (pero no podemos rechazarla o probar lo contrario).

#------------Formular hipótesis de una cola---------------
#python por defecto hace hipotesis de dos colas, si quieres una unilateral, divide el valor p entre dos
from scipy import stats as st
import pandas as pd
screens = pd.Series([4, 2, 4, 5, 5, 4, 2, 3, 3, 5, 2, 5, 2, 2, 2, 3, 3, 4, 8, 3, 4, 3, 5, 5, 4, 2, 5, 2, 3, 7, 5, 5, 6,  5, 3, 4, 3, 6, 3, 4, 4, 3, 5, 4, 4, 8, 4, 7, 4, 5, 5, 3, 4, 6, 7, 2, 3, 6, 5, 6, 4, 4, 3, 4, 6, 4, 4, 6, 2, 6, 5, 3, 3, 3, 4, 5, 3, 5, 5, 4, 3, 3, 3, 1, 5, 4, 3, 4, 6, 3, 1, 3, 2, 7, 3, 6, 6, 6, 5, 5])
prev_screens_value = 4.867 # número promedio de bloques vistos
alpha = 0.05  # nivel de significación
results = st.ttest_1samp(screens, prev_screens_value)
print('valor-p: ', results.pvalue / 2)# prueba unilateral: el valor p se divide en dos
# prueba unilateral a la izquierda:
# rechaza la hipótesis solo si la media muestral es significativamente menor que el valor propuesto
if (results.pvalue / 2 < alpha) and (screens.mean() < prev_screens_value):#comparamos el valor p con el umbral y la media muestral con el valor propuesto
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

#Si la prueba fuera unilateral a la derecha con la hipótesis alternativa “El valor observado es mayor que el predicho”, uno de los signos “<” cambiaría a “>”. Las últimas líneas de código se verían así:
if (results.pvalue / 2 < alpha) and (screens.mean() > prev_screens_value):
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")

from scipy import stats as st
import numpy as np
import pandas as pd
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
#Dos muestras independientes
#equal_var: varianza igual es un parámetro opcional que especifica si las varianzas de las poblaciones deben considerarse iguales, True o False
#True y la varianza de cada muestra se estimará a partir del dataset combinado de las dos muestras
from scipy import stats as st
import numpy as np
sample_1 = [3071, 3636, 3454, 3151, 2185, 3259, 1727, 2263, 2015, 
            2582, 4815, 633, 3186, 887, 2028, 3589, 2564, 1422, 1785, 
            3180, 1770, 2716, 2546, 1848, 4644, 3134, 475, 2686, 
            1838, 3352]
sample_2 = [1211, 1228, 2157, 3699, 600, 1898, 1688, 1420, 5048, 3007, 
            509, 3777, 5583, 3949, 121, 1674, 4300, 1338, 3066, 
            3562, 1010, 2311, 462, 863, 2021, 528, 1849, 255, 
            1740, 2596]
alpha = 0.05  # el nivel de significancia estadística crítica
results = st.ttest_ind(sample_1, sample_2) # si el valor p es menor que alpha, rechazamos la hipótesis
print('valor p: ', results.pvalue) 
if results.pvalue < alpha: # comparar el valor p con el umbral
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


from scipy import stats as st
import numpy as np
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


from scipy import stats as st
import numpy as np
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
#La varianza mide qué tan dispersos o separados están los datos de su promedio. 
# Grupo A: 8, 8, 8, 8, 8 (promedio = 8) Varianza baja (todos muy similares)
# Grupo B: 2, 6, 8, 12, 12 (promedio = 8) Varianza alta (muy dispersos)

# Puedes calcularla así:
import numpy as np
varianza_logpass = np.var(time_on_site_logpass)
varianza_social = np.var(time_on_site_social)

#-----------Hipótesis sobre la igualdad de las medias de muestras emparejadas---------------
#Dos muestras relacionadas/pareadas
#muestras emparejadas, lo que significa que medimos una variable dos veces para cada cliente, antes y después de los cambios.
from scipy import stats as st
import numpy as np
before = [157, 114, 152, 355, 155, 513, 299, 268, 164, 320, #deben tener el mismo tamaño
                    192, 262, 506, 240, 364, 179, 246, 427, 187, 431, 
                    320, 193, 313, 347, 312, 92, 177, 225, 242, 312]
after = [282, 220, 162, 226, 296, 479, 248, 322, 298, 418, 
                 552, 246, 251, 404, 368, 484, 358, 264, 359, 410, 
                 382, 350, 406, 416, 438, 364, 283, 314, 420, 218]
alpha = 0.05  # el nivel de significancia estadística crítica
results = st.ttest_rel(before, after)
print('valor p: ', results.pvalue)
if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula") #ha habido un cambio 
else:
    print("No podemos rechazar la hipótesis nula") #no ha habido un cambio

#Seleccionamos una prueba unilateral debido a la palabra "más". Si no sabemos la dirección del cambio, entonces utilizaremos la prueba bilateral.
from scipy import stats as st
import numpy as np
import pandas as pd
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
#una técnica de muestreo donde dividimos la población en grupos más pequeños llamados estratos (que comparten características similares), y luego tomamos muestras aleatorias de cada estrato.

#-----------Teorema del límite central--------------
#Si tomas muchas muestras de cualquier población y calculas la media de cada muestra, esas medias se distribuirán de forma normal (curva de campana), sin importar cómo se vea la población original.

#-----------¿Qué es el nivel de significancia?--------------
#es el umbral que establecemos antes de realizar una prueba estadística para decidir si rechazamos o no la hipótesis nula.

#-----------Función de densidad de probabilidad--------------
#una curva que describe todas las posibilidades de que ocurra un evento; Dónde es más probable que ocurran ciertos valores; Dónde es menos probable que ocurran otros valores

#-----------¿Qué es una prueba t?--------------
#es un método estadístico que te ayuda a determinar si existe una diferencia significativa
#tiene "colas más gruesas", lo que la hace más apropiada cuando trabajas con muestras pequeñas (menos de 30 observaciones) o cuando no conoces la desviación estándar de toda la población.
# la hipótesis nula siempre implica la ausencia de diferencias.
# Si el valor p es mayor que el valor alfa ⇒ no podemos rechazar H₀.

