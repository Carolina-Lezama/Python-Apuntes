#--------------------------MÉTRICAS DE PRODUCTO-----------------------------------
miden el impacto en el negocio y la experiencia del usuario
1. Tasa de retención de clientes: ¿Cuántos clientes logramos mantener?
2. Costo de adquisición vs retención: ¿Es más barato retener que conseguir nuevos clientes o mantener los actuales?
3. Revenue por cliente: ¿Cuánto dinero genera cada cliente?
4. Satisfacción del cliente: ¿Los clientes están contentos?
    Se enfocan en resultados de negocio

#--------------------------MÉTRICAS DE ML (Machine Learning)----------------------------------
evalúan el rendimiento técnico del modelo
F1-Score: Balance entre precisión y recall
AUC-ROC: Capacidad de distinguir entre clases
Accuracy: Porcentaje de predicciones correctas
Precision/Recall: Qué tan exacto y completo es el modelo
Se enfocan en rendimiento algorítmico

#--------------------------ANALOGÍA SIMPLE----------------------------------
Métricas ML = ¿Qué tan bien funciona el motor del auto?
Métricas de Producto = ¿Qué tan rápido llego a mi destino y cuánto combustible gasto?

#--------------------------Ingresos, coste de los bienes vendidos y margen-----------------------------------
Ingresos:
cantidad de dinero que los clientes pagan a la empresa. 
Ejemplo:
un artículo tiene un precio de 10 dólares. Si se venden 10, los ingresos serán 10×10=100 dólares.

Coste de la materia consumida (coste de los bienes vendidos):
dinero pagado por la empresa por la compra del producto (material para la produccion).
Ejemplo:
para un taxista, el coste de la gasolina está incluido en el coste del viaje de un pasajero

Beneficio bruto:
Beneficio bruto = Ingresos  − Coste de los bienes
                    1000000 − 300000 = 700000
La empresa compró poco y vendió mucho (beneficio bruto positivo)
Si el margen bruto es negativo, ya que indica que la empresa vende menos de lo que compra

Margen bruto:
Margen de beneficio bruto = (Beneficio bruto/Ingresos) x 100%
                                    ( 700000/1000000 ) = 70%

#--------------------------Beneficios y gastos operativos----------------------------------
Aquí es donde se incluyen todos los gastos relacionados con las actividades comerciales de la empresa: salarios de los empleados, alquiler de la oficina, servicios públicos, marketing y otros gastos generales.

Beneficio operativo:
Beneficio operativo = Beneficio bruto - Gastos operativos
                            700000    -    200000   = 500000

El beneficio operativo ayuda a averiguar cuánto gana una empresa gracias a sus actividades comerciales.
Salvo contadas excepciones, el beneficio operativo se correlaciona con el beneficio neto: cuanto mayor sea el beneficio operativo, mayor será el beneficio neto

Margen de beneficio operativo = (Beneficio operativo / Ingresos) x 100%
                                (500000              / 1000000)  ×100%   = 50%
Se trata de la parte de los ingresos que queda en la empresa tras deducir el coste relacionado con las actividades principales

#--------------------------Beneficio neto----------------------------------
Beneficio neto = Beneficio operativo - Impuestos y prestamos(este normalmente es completo o intereses)
1000000 x 0.06 = 60000 #impuestos
500000 - 60000 = 440000 #beneficio neto

Un beneficio neto negativo se denomina pérdida neta. Muestra que la empresa no consiguió ganar dinero.

#----------------------Rendimiento sobre la inversión - ROI (return on investment)----------------------------------

ROI= (Beneficio neto - Inversión total) / Inversión
ROI= (      44000          -300000)     /  300000    = 0.8533 = -85.33%

#--------------------------Conversión----------------------------------
El proceso de venta puede describirse así: visitante → venta → comprador.
la conversión: muestra el porcentaje de usuarios que completaron una acción determinada

Si 1000 personas visitaron la tienda y 10 de ellas compraron algo: la tasa de conversión de usuarios a clientes es de 10/1000 = 0.01 o 1 %.

#--------------------------Embudos----------------------------------
El camino que sigue el usuario para comprar un producto
La proporción de personas que pasan de una etapa a otra

Ejemplo:
1. Accedió a la página de inicio de la tienda
2. Entró en la página del producto
3. Añadió un artículo al carrito
4. Pasó a la página de pago
5. Pagó el pedido

#el grafico se asemeja a un embudo  
Etapa	            Cantidad	Conversión total	Conversión a la siguiente etapa
Página principal	10000	        100%	                    80% (8,000 van a producto)
Página del producto	8000	        80%	                        37,5% (3,000 añaden al carrito)
Añadido a la cesta	3000	        30%	                        33,3%  (1,000 van a pago)
Página de pago	    1000	        10%	                        10%  (100 completan compra)
Pedido	            100	            1%	                          -

#--------------------------Métricas online y offline---------------------------------
¿QUÉ ES LA DESCOMPOSICIÓN DE MÉTRICAS?
dividir una métrica compleja en  componentes más simples para entender mejor qué factores la afectan

INGRESOS POR DÍA
├── Número de visitantes por día
├── Conversión (% que compra)
└── Importe promedio de compra
    ├── Costo de productos originales (carrito inicial)
    └── Costo de productos recomendados (por el modelo ML)

Sin descomposición: "Los ingresos bajaron 10%"
Con descomposición: 
Visitantes: ↑ +5%
Conversión: ↓ -8% ← ¡Aquí está el problema! un problema desencadeno al siguiente, sin conversion bajan las ventas
Recibo promedio: ↓ -7%

Ejemplo de descomposición:
Usuarios por día: 1,000 visitantes
Conversión: 5% (50 personas compran)
Recibo promedio: $100
Productos originales: $80
Productos recomendados: $20
Ingresos = 1,000 × 0.05 × $100 = $5,000/día

métrica online: No puedes calcularla con datos del historial.
Son como "exámenes reales" con usuarios verdaderos usando tu sistema ahora mismo.
- Tu modelo recomienda productos a usuarios reales HOY
- Mides: ¿Cuántos usuarios realmente compraron lo recomendado?
- Calculas: ingresos generados por las recomendaciones

Ingresos por recomendaciones: $50,000 este mes
Tasa de conversión: 15% de usuarios compraron lo recomendado

métricas offline: Se calculan con los datos del historial.
Son como "exámenes de práctica" que haces con datos del pasado, antes de lanzar tu modelo
- Tienes datos de clientes que compraron en 2023
- Tu modelo predice: "A María le gustará esta blusa azul"
- Calculas precisión, recall, F1-score usando estos datos del pasado

Precisión: De 100 auriculares recomendados, 70 fueron comprados = 70%
Recall: De 80 auriculares que el cliente quería, recomendaste 60 = 75%

#--------------------------Establecer objetivos---------------------------------
Al seleccionar una métrica, el equipo de producto determina cómo aumentar su valor en función del objetivo comercial.

Situación:
- Tienes un modelo de recomendaciones EXCELENTE (precisión 95%)
- Pero... el botón de "Comprar" está roto en tu página web
- Resultado: Cero ventas, cero ingresos

Lección: No importa qué tan bueno sea tu modelo si hay problemas en otras partes del negocio.

¿Cómo establecer objetivos correctos?
Paso 1: Entiende el problema REAL del negocio
    ¿El problema es realmente el modelo de ML?
    ¿O hay otros obstáculos (interfaz, proceso de pago, etc.)?
Paso 2: Elige la métrica correcta según tu objetivo
Ejemplo práctico: Sistema de recomendaciones de Netflix
Si tu objetivo es: Que los usuarios vean MÁS contenido
    Métrica importante: Recall (encontrar todas las películas que le podrían gustar)
    Menos importante: Precisión
Si tu objetivo es: Que los usuarios confíen en tus recomendaciones
    Métrica importante: Precisión (que cada recomendación sea muy buena)
    Menos importante: Recall
Paso 3: Investiga el comportamiento del usuario
    Encuestas: "¿Por qué no compraste el producto recomendado?"
    Análisis de datos: ¿Dónde abandonan los usuarios el proceso?

#--------------------------Planificación de la implementación---------------------------------
¿Debería la empresa implementar un nuevo servicio o no?

Pruebas A/B o split testing: 
Es una técnica de comprobación de hipótesis, ayudan a controlar el impacto que provocan los cambios de servicio o producto sobre los usuarios

la población se divide en grupo de control (A) y grupo experimental (B)
El grupo A utiliza el servicio habitual sin cambios. 
El grupo B utiliza la nueva versión, que es la que tenemos que probar.
El experimento dura un periodo determinado
El objetivo es recopilar datos sobre el comportamiento de los visitantes en ambos grupos
Si la métrica clave en el grupo experimental mejora en comparación con el grupo de control, entonces se implementará la nueva funcionalidad
LOS GRUPOS NO SE ALTERAN DURANTE EL EXPERIMENTO hasta que finalice el experimento

Antes de las pruebas A/B, a menudo se utiliza la prueba A/A(comprobación de validez)
los visitantes se dividen en dos grupos de control que se exponen a la misma versión del servicio
La métrica clave debe coincidir en ambos grupos

Un experimento demasiado largo ralentiza la implementación de la nueva funcionalidad. 
En cambio, es posible que un experimento demasiado corto no proporcione resultados fiables.

#--------------------------Duración de las pruebas A/B---------------------------------
los usuarios necesitan tiempo para acostumbrarse a los cambios
Una vez que se hayan acostumbrado por completo, podremos evaluar con certeza si el experimento ha sido un éxito. 

Cuantos más datos tengamos, menor será la probabilidad de error al comprobar las hipótesis

Las pruebas A/B sufren el llamado peeking problem:
    consiste en que el resultado final se distorsiona cuando se añaden nuevos datos al principio del experimento

Imagina que estás horneando un pastel y cada 5 minutos abres el horno para ver si ya está listo.
Cada vez que lo haces, cambias las condiciones del horno y afectas el resultado final. 
El peeking problem es similar: cuando revisas los resultados de tu prueba A/B constantemente durante los primeros días, puedes llegar a conclusiones incorrectas.

Cuando los datos son escasos, cualquier valor atípico (como un usuario muy activo o muy inactivo) puede distorsionar completamente los resultados.
La dispersión tiende a ser mayor cuando el número de observaciones es escaso
cuando tenemos un gran número de observaciones, el impacto de los valores atípicos se reduce

sin una validación adecuada, podemos alcanzar nuestro umbral de valor p de forma incorrecta y, por tanto, rechazar erróneamente la hipótesis nula.

la significancia estadística se alcanza durante los primeros días de la prueba
la decisión tomada al alcanzar la significación estadística sería incorrecta.

1.recolectar observacion x
2.calcular metricas y valor p
3.si valor p < umbral entonces finalizar experimento y los grupos son diferentes
4.si no los grupos no difieren

#-------------------------Calculadora en linea para la duración de la prueba A/B ---------------------------------
https://vwo.com/tools/ab-duration-calculator/

Average number of daily visitors who will participate in the test (control + variation)
10000

Estimated existing conversion rate (%)
7%

Minimum improvement in conversion rate you want to detect (%)
7.1%

Number of variations/combinations (including control)
2

#-------------------------Comparacion de las medias--------------------------------
¿Qué describe el comportamiento de todos los usuarios en la prueba A/B? 
¡El valor medio de la métrica! 
Podría ser el tiempo medio de navegación en el sitio web, el importe medio de las compras o el número medio de usuarios únicos.

El promedio convierte muchos datos individuales en UN SOLO número que representa a todo el grupo.
No podemos predecir el valor de cada observación con exactitud absoluta, pero podemos estimarlo utilizando métodos estadísticos.

H₀(nula) dice: la nueva funcionalidad no mejora las métricas. a 
H₁(alternativa) será: la nueva funcionalidad mejora las métricas.

tipos de errores en la comprobacion de hipótesis:
Error tipo I (falso positivo): Rechazar H₀ cuando es verdadera.
Error tipo II (falso negativo): No rechazar H₀ cuando es falsa.

Para aceptar o rechazar la hipótesis nula, calculemos el nivel de significación, también conocido como valor p (valor de probabilidad).
Los valores medios se comparan utilizando los métodos de prueba de hipótesis unilateral
Si la distribución de los datos se aproxima a la normalidad (no hay valores atípicos significativos en los datos), se utiliza la prueba estándar para comparar las medias

import pandas as pd
from scipy import stats as st
sample_before = pd.Series([
    436, 397, 433, 412, 367, 353, 440, 375, 414, 
    410, 434, 356, 377, 403, 434, 377, 437, 383,
    388, 412, 350, 392, 354, 362, 392, 441, 371, 
    350, 364, 449, 413, 401, 382, 445, 366, 435,
    442, 413, 386, 390, 350, 364, 418, 369, 369, 
    368, 429, 388, 397, 393, 373, 438, 385, 365,
    447, 408, 379, 411, 358, 368, 442, 366, 431,
    400, 449, 422, 423, 427, 361, 354])
sample_after = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])
print("La media de antes:", sample_before.mean())
print("La media de después:", sample_after.mean())
# nivel crítico de significación; la hipótesis se rechaza si el valor p es menor que ese
alpha = 0.05
results = st.ttest_ind(sample_before, sample_after)
# prueba unilateral (de una cola): el valor p será la mitad
pvalue = results.pvalue / 2
print('p-value: ', pvalue)
if pvalue < alpha:
    print(
        "La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente"
    )
else:
    print(
        "La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente"
    )

#-------------------------Intervalo de confianza--------------------------------
¿qué podemos hacer para averiguar la exactitud de una media?

rango numerico del parámetro poblacional de interés, con una probabilidad predeterminada.
El parámetro es desconocido, pero podemos estimarlo a partir de la muestra

Imagina que tienes una caja cerrada y quieres saber cuánto pesa exactamente. No puedes abrirla, pero puedes:
Levantarla varias veces (tomar muestras)
Estimar su peso basándote en esas "pruebas"

El problema: Cada vez que la levantas, tu estimación varía un poco:
- Primera vez: "Pesa como 4.7 kg"
- Segunda vez: "Pesa como 4.9 kg"  
- Tercera vez: "Pesa como 4.6 kg"

¿Cuál es el peso REAL? No lo sabes exactamente, pero puedes decir:
"Estoy 99% seguro de que el peso real está entre 4.5 kg y 5.0 kg"
¡Eso es un intervalo de confianza del 99%!

### "Un parámetro poblacional de interés"
- Parámetro poblacional: El valor REAL que quieres conocer
- En tu caso: El promedio real de compra de TODOS los usuarios

### "Con una probabilidad predeterminada"
- 99% de confianza: Si repites este proceso 100 veces, en 99 ocasiones el valor real estará dentro del intervalo

### "El parámetro es desconocido, pero podemos estimarlo"
- Desconocido: No puedes medir a TODOS los usuarios
- Estimación: Usas tu muestra para aproximarte

Imagina que lanzas dardos a una diana:

El centro de la diana: El valor real (desconocido)
Tus lanzamientos: Tus muestras
El área alrededor del centro: Tu intervalo de confianza
Intervalo del 99%: "Estoy 99% seguro de que mi dardo cayó en esta área grande"
Intervalo del 50%: "Estoy 50% seguro de que mi dardo cayó en esta área pequeña"
El intervalo de confianza  dice: "Mi estimación sugiere que el valor real (fijo) probablemente está entre 300 y 500"

Al calcular el intervalo de confianza, normalmente descartamos la misma cantidad de valores de cada uno de sus extremos. 
Por ejemplo, para construir un intervalo de confianza del 99 %, tenemos que descartar el 1 % de los valores extremos, lo cual supone el 0.5 % de los valores más grandes y el 0.5 % de los más pequeños.

El intervalo de confianza no es solo un rango de valores aleatorios.
La causa de la variabilidad radica en el hecho de que el número es desconocido y que se calcula a partir de la muestra. 
El carácter aleatorio de la muestra introduce aleatoriedad en la estimación. 
El intervalo de confianza mide la confianza en dicha estimación.

El percentil de 2.5 % y el percentil de 97.5 %
¡Correcto! 97.5 % - 2.5 % = 95 %.

#-------------------------Calculo de intervalo de confianza--------------------------------
El teorema del límite central dice que todas las medias de todas las muestras posibles con un tamaño n se distribuyen normalmente alrededor de la verdadera media poblacional. 
"Alrededor" significa que la media de esta distribución de todas las medias muestrales será igual a la verdadera media poblacional. 
La varianza será igual a la varianza poblacional dividida entre n (el tamaño de la muestra).

La desviación estándar de esta distribución se denomina error estándar
Cuanto mayor sea el tamaño de la muestra, menor será el error estándar
Cuanto mayor sea la muestra, más precisa será la estimación.

Para calcular el error estándar, utilizamos la varianza poblacional, pero la desconocemos al igual que la media poblacional. 
La estimamos a partir de la muestra.

import pandas as pd
from scipy import stats as st
sample = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])
print('Media:', sample.mean())
confidence_interval = st.t.interval(
    0.95, len(sample)-1,sample.mean(),sample.sem() )
print('Intervalo de confianza del 95 %:', confidence_interval)

confidence : nivel de confianza
df: número de grados de libertad (igual a n - 1)
loc (de localización): la distribución media igual a la estimación media.
scale: el error estándar de la distribución igual a la estimación del error estándar.

Resultado:
    Media: 470.5285714285714
    Intervalo de confianza del 95 %: (463.357753651609, 477.6993892055338)

#-------------------------Bootstrapping--------------------------------
expresión en inglés to pull oneself up by one’s bootstraps
Esto significa que algo que parece imposible de conseguir sin ayuda externa, en realidad sí se puede hacer.

El bootstrapping funciona con cualquier distribución de datos.
El bootstrapping forma nuevas muestras con base en una muestra fuente.

El bootstrapping es aplicable a una amplia gama de muestras y es particularmente útil cuando:
    Los datos no están distribuidos normalmente.
    No hay pruebas estadísticas paramétricas adecuadas para el valor objetivo.
    El bootstrapping se puede aplicar en casos en los que las pruebas paramétricas son inadecuadas y es valioso para estimar intervalos de confianza independientemente de la distribución adyacente.
    La atención se centra en situaciones en las que los supuestos de distribución normal fallan o en las que las pruebas estadísticas no están bien definidas.

#-------------------------Bootstrapping para el intervalo de confianza--------------------------------
permite estimar un rango de valores plausibles para el percentil de el parámetro poblacional de interés, basándose únicamente en los datos de la muestra
Esto es crucial para la toma de decisiones porque reconoce la incertidumbre inherente al trabajar con una muestra de datos.
El Bootstrapping proporciona una forma de cuantificar esta incertidumbre al calcular un intervalo de confianza.  

El intervalo de confianza del 90% indica que si se repitiera este proceso de prueba muchas veces, el 90% de los intervalos calculados contendrían el verdadero tiempo de espera del percentil 99.

#obtener muestras aleatorias
from numpy.random import RandomState
state = RandomState(12345)
for i in range(5):
    # extrae un elemento aleatorio de la muestra 1
    print(data.sample(1, random_state=state))

Ejemplo resultado:
    69    13.68
    dtype: float64
    27    8.44
    dtype: float64
    36    10.54
    dtype: float64
    84    18.44
    dtype: float64
    59    15.76
    dtype: float64

el mismo elemento puede caer en una submuestra varias veces
example_data = pd.Series([1, 2, 3, 4, 5])
print(example_data.sample(frac=1, replace=True, random_state=state))

frac = fracción del dataset original
frac=0.15: Quieres el 15% del total
frac=1: Quieres el 100% de los valores 

import pandas as pd
import numpy as np
data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])
state = np.random.RandomState(12345)
for i in range(10):
    subsample = data.sample(frac=0.15, replace=True, random_state=state)
    print(subsample.quantile(0.99))

La función quantile() calcula el percentil/cuantil de una Serie de datos.
un valor entre 0 y 1 que representa el percentil deseado.

# Calcular varios percentiles
subsample.quantile([0.25, 0.5, 0.75, 0.99])

# Obtener exactamente 15 elementos
muestra = data.sample(n=15)
print(len(muestra))  # → 15

# Obtener 15% del dataset
muestra = data.sample(frac=0.15)
print(len(muestra))  # → 15 elementos

import pandas as pd
import numpy as np
data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])
state = np.random.RandomState(12345)
values = []
for i in range(1000):
    subsample = data.sample(frac=0.15, replace=True, random_state=state)
    values.append(subsample.quantile(0.99)) # percentil 99
values = pd.Series(values)
    
lower = values.quantile(0.05) #confianza del 90%
upper = values.quantile(0.95)
print(lower)
print(upper)

#-------------------------¿Qué es un percentil?--------------------------------
Analogía: Las calificaciones de tu clase
Imagina que hay 100 estudiantes en tu clase y estos son sus puntajes en un examen:
Puntajes ordenados de menor a mayor:
- Estudiante más bajo: 45 puntos
- Estudiante en posición 50: 78 puntos  
- Estudiante en posición 99: 95 puntos
- Estudiante más alto: 98 puntos
¿Qué significa percentil 99%?
El percentil 99 = 95 puntos
Esto significa: "El 99% de los estudiantes sacó 95 puntos o menos"
Solo el 1% de los estudiantes sacó más de 95 puntos.

Analogía: Velocidad de carga de un sitio web
Si mides los tiempos de carga:
Promedio: 2.3 segundos
Percentil 99: 8.7 segundos
Esto te dice: "El 99% de los usuarios experimentan tiempos de carga de 8.7 segundos o menos"
¡Solo el 1% más lento experimenta tiempos mayores a 8.7 segundos!

subsample.quantile(0.99)

#------------------¿Y qué es el "intervalo de confianza del 90% para el percentil 99%"?--------------------------------
Significa: "Estoy 90% confiado de que el percentil 99 real de toda la población está dentro de este rango"

#-------------------------Bootstrapping para el analisis de pruebas A/B-------------------------------
El bootstrapping no elimina el concepto de la hipótesis estadística, pero ofrece otra forma de probarlas.

La hipotesis dependen de:
1. Lo que quieres probar
2. El tipo de comparación
3. La dirección del efecto esperado

H₀: D = 0    (No hay diferencia)
H₁: D > 0    (B es mayor que A, hay aumento)  
Donde D = promedio_B - promedio_A

Ejemplo 1: ¿Ha cambiado algo? (bidireccional)
H₀: μ = 100     (La media es 100)
H₁: μ ≠ 100     (La media es diferente de 100)  ← Interesa cualquier cambio

Ejemplo 2: ¿Ha disminuido? (unidireccional)
H₀: μ ≥ 50      (La media es mayor o igual a 50)
H₁: μ < 50      (La media es menor que 50) ← Solo interesa si aumentó o disminuyo

Ejemplo 3: Comparando dos medicamentos
H₀: μ_medicamento_A = μ_medicamento_B    (Son iguales)
H₁: μ_medicamento_A ≠ μ_medicamento_B    (Son diferentes)

Ejemplo 4: ¿Es efectivo un tratamiento?
H₀: μ_después ≤ μ_antes    (No mejora o empeora)
H₁: μ_después > μ_antes    (Mejora)

¿Qué quiero demostrar específicamente?
Aumento → H₁: D > 0
Disminución → H₁: D < 0
Cualquier cambio → H₁: D ≠ 0

¿Tengo una dirección esperada?
Sí → Prueba de una cola
No → Prueba de dos colas

La hipótesis nula consiste en que no hay diferencia entre los  promedio de los dos grupos.
La hipótesis alternativa dice que el  promedio de los grupos es mayor/menor en el grupo experimental

Si el valor p está por debajo del umbral de significación estadística, podemos rechazar H0 a favor de HA

si calculamos simplemente la diferencia de forma aritmética, es probable que esta varíe ligeramente a causa de la aleatoriedad de las muestras

pasos:
1. Calcular la diferencia observada entre los promedios de los dos grupos
    Dada la hipótesis H0 suponemos que los grupos son iguales.
2. Combinar los datos de ambos grupos en un solo conjunto (concatenamos las observaciones)
3. Generar muchas muestras de bootstrap a partir del conjunto combinado
    seleccionando dos muestras (una para representar al grupo A y otra para representar al B)
    Para cada experimento simulado, se calcula la diferencia del importe promedio de compra para esa iteración(D1)

valor p: puede estimarse entonces como la razón entre el número de veces cuando Di no fue inferior a D y el número de experimentos simulados.

"disparidad" se refiere a la diferencia entre los promedios de los dos grupos.

D = Promedio del grupo B - Promedio del grupo A

Si el valor p es inferior a un determinado umbral (normalmente, 0,05), podemos rechazar la hipótesis nula y decir que existe una diferencia significativa entre los dos grupos.


import pandas as pd
import numpy as np
samples_A = pd.Series([ # datos del grupo de control A
     98.24,  97.77,  95.56,  99.49, 101.4 , 105.35,  95.83,  93.02,
    101.37,  95.66,  98.34, 100.75, 104.93,  97.  ,  95.46, 100.03,
    102.34,  98.23,  97.05,  97.76,  98.63,  98.82,  99.51,  99.31,
     98.58,  96.84,  93.71, 101.38, 100.6 , 103.68, 104.78, 101.51,
    100.89, 102.27,  99.87,  94.83,  95.95, 105.2 ,  97.  ,  95.54,
     98.38,  99.81, 103.34, 101.14, 102.19,  94.77,  94.74,  99.56,
    102.  , 100.95, 102.19, 103.75, 103.65,  95.07, 103.53, 100.42,
     98.09,  94.86, 101.47, 103.07, 100.15, 100.32, 100.89, 101.23,
     95.95, 103.69, 100.09,  96.28,  96.11,  97.63,  99.45, 100.81,
    102.18,  94.92,  98.89, 101.48, 101.29,  94.43, 101.55,  95.85,
    100.16,  97.49, 105.17, 104.83, 101.9 , 100.56, 104.91,  94.17,
    103.48, 100.55, 102.66, 100.62,  96.93, 102.67, 101.27,  98.56,
    102.41, 100.69,  99.67, 100.99])
samples_B = pd.Series([ # datos del grupo experimental B
    101.67, 102.27,  97.01, 103.46, 100.76, 101.19,  99.11,  97.59,
    101.01, 101.45,  94.8 , 101.55,  96.38,  99.03, 102.83,  97.32,
     98.25,  97.17, 101.1 , 102.57, 104.59, 105.63,  98.93, 103.87,
     98.48, 101.14, 102.24,  98.55, 105.61, 100.06,  99.  , 102.53,
    101.56, 102.68, 103.26,  96.62,  99.48, 107.6 ,  99.87, 103.58,
    105.05, 105.69,  94.52,  99.51,  99.81,  99.44,  97.35, 102.97,
     99.77,  99.59, 102.12, 104.29,  98.31,  98.83,  96.83,  99.2 ,
     97.88, 102.34, 102.04,  99.88,  99.69, 103.43, 100.71,  92.71,
     99.99,  99.39,  99.19,  99.29, 100.34, 101.08, 100.29,  93.83,
    103.63,  98.88, 105.36, 101.82, 100.86, 100.75,  99.4 ,  95.37,
    107.96,  97.69, 102.17,  99.41,  98.97,  97.96,  98.31,  97.09,
    103.92, 100.98, 102.76,  98.24,  97.  ,  98.99, 103.54,  99.72,
    101.62, 100.62, 102.79, 104.19])
AB_difference = samples_B.mean() - samples_A.mean() # diferencia real entre las medias de los grupos
print("Diferencia entre los importes promedios de compra:", AB_difference)
alpha = 0.05
state = np.random.RandomState(12345)
bootstrap_samples = 1000 # número de experimentos simulados
count = 0 #Número de veces que el bootstrapping produjo una diferencia igual o mayor a AB_difference
for i in range(bootstrap_samples):# calcula cuántas veces excederá la diferencia entre las medias
    united_samples = pd.concat([samples_A, samples_B]) # combina las muestras de ambos grupos
    subsample = united_samples.sample(frac=1, replace=True, random_state=state) # crea una muestra de un tamaño igual al conjunto combinado con valores que pueden repetirse
    subsample_A = subsample[:len(samples_A)] #toma los primeros 100 de subsample
    subsample_B = subsample[len(samples_A):] #Toma el resto de elementos desde la posición len(samples_A) hasta el final
    bootstrap_difference = subsample_B.mean() - subsample_A.mean() # diferencia entre las medias de las dos submuestras
    if bootstrap_difference >= AB_difference: #cada que sea mayor o igual a la diferencia real, incrementa el contador
        count += 1
pvalue = 1. * count / bootstrap_samples # Convierte el resultado a número decimal (float) / numero de experimentos simulados, obteniendo el valor p
    #Ejemplo:  pvalue = 1. * 50 / 1000 = 0.05
print('p-value =', pvalue)
if pvalue < alpha:
    print("La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente")
else:
    print("La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente")

¿Qué representa esta disparidad?
### Si AB_difference > 0:
- El grupo B tiene un promedio mayor que el grupo A
- Interpretación: "El importe promedio de compra aumentó"

### Si AB_difference < 0:
- El grupo B tiene un promedio menor que el grupo A  
- Interpretación: "El importe promedio de compra disminuyó"

### Si AB_difference = 0:
- Ambos grupos tienen el mismo promedio
- Interpretación: "No hay diferencia en el importe promedio"

#Ejemplo
### Supongamos:
- Grupo A (control): Promedio = 99.5€
- Grupo B (experimental): Promedio = 101.2€
Disparidad: AB_difference = 101.2 - 99.5 = 1.7€
Interpretación: "El importe promedio de compra aumentó en 1.7€"

¿La diferencia es real o solo casualidad?
Escenario 1: La diferencia es solo casualidad
Es como si ambas cajas tuvieran exactamente el mismo tipo de caramelos, pero por pura suerte sacaste caramelos más pesados de una caja.
En este caso, si mezclas todos los caramelos de ambas cajas y los redistribuyes aleatoriamente en dos nuevas cajas, es muy probable que:
Muchas veces obtengas diferencias iguales o mayores a D
El proceso de redistribución (bootstrapping) no tendrá problemas para "recrear" esa diferencia

Escenario 2: La diferencia es genuina
Es como si una caja realmente tuviera caramelos más pesados que la otra (por ejemplo, unos tienen chocolate y otros no).
En este caso, cuando mezclas todos los caramelos y los redistribuyes:
Será muy difícil recrear la diferencia original D
El bootstrapping "luchará" para producir diferencias tan grandes
¿Por qué? Porque no puede crear información nueva - solo puede redistribuir lo que ya tienes

El bootstrapping te ayuda a determinar si la diferencia en los importes de compra entre los grupos A y B es:
    Casualidad: Si el p-value es alto (muchas redistribuciones recrean la diferencia)
    Genuina: Si el p-value es bajo (pocas redistribuciones recrean la diferencia)

Interpretación:
### Si p-value es bajo (ej: 0.01):
- Solo 1% de probabilidad de que sea casualidad
- Conclusión: La diferencia probablemente es real

### Si p-value es alto (ej: 0.30):
- 30% de probabilidad de que sea casualidad
- Conclusión: La diferencia podría ser solo suerte

En tu código:
Si pvalue < alpha (0.05), rechazas H₀ y concluyes que sí hay aumento real.

#-------------------------Bootstrapping para modelos------------------------------
import pandas as pd
target = pd.Series([1,   1,   0,   0,  1,    0])
probab = pd.Series([0.2, 0.9, 0.8, 0.3, 0.5, 0.1])
def revenue(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    return 10 * selected.sum() 
res = revenue(target, probab, 3)
print(res)

import pandas as pd
import numpy as np

target = pd.read_csv('/datasets/eng_target.csv')['0']
probabilities = pd.read_csv('/datasets/eng_probabilites.csv')['0']

def revenue(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    return 10 * selected.sum()

state = np.random.RandomState(12345)
    
values = []
for i in range(1000):
    target_subsample = target.sample(n=25, replace=True, random_state=state)
    probs_subsample = probabilities[target_subsample.index]
    
    values.append(revenue(target_subsample, probs_subsample, 10))

values = pd.Series(values)
lower = values.quantile(0.01)

mean = values.mean()
print("Ingresos promedio:", mean)
print("Cuantil del 1 %:", lower)

#Resultado:
Ingresos promedio: 91.67
Cuantil del 1 %: 60.0


































#-------------------------Fuentes de datos------------------------------
Crawler (rastreador):
- Es como un "robot" que navega automáticamente por páginas web
- Sigue enlaces de una página a otra, como una araña en su telaraña
- Descarga todas las páginas de un sitio web de forma sistemática

Scraper (extractor):
- Es el software que "extrae" o "raspa" la información específica de las páginas descargadas
- Toma el HTML de la página y extrae solo los datos que necesitas

Crawler: Navega automáticamente por Twitter y descarga miles de tweets
Scraper: Extrae solo el texto de los tweets, ignorando imágenes, enlaces, etc.
Resultado: Tienes una base de datos con miles de comentarios para entrenar tu modelo

¿Por qué es útil?
Vocabulario real: Los datos de redes sociales contienen lenguaje coloquial, jerga, emojis
Datos actuales: Información en tiempo real sobre opiniones y tendencias
Gran volumen: Puedes obtener millones de ejemplos

Consideraciones importantes:
Respetar los términos de uso de cada sitio web
Algunos sitios tienen APIs oficiales (como Twitter API)
Considerar aspectos legales y éticos

https://archive.ics.uci.edu/
https://data.gov/about/
https://data.fivethirtyeight.com/

#-------------------------Etiquetado de datos------------------------------
Para obtener un conjunto de entrenamiento, debemos realizar etiquetado de datos o anotación de datos.
una habilidad fundamental cuando no tienes suficientes datos para tu proyecto de machine learning.

- La empresa no tiene una base de datos interna con los datos que necesitas
- Los datos que existen no están "etiquetados" (sin valores objetivo)
- Necesitas datos externos para complementar tu análisis

Soluciones:
    ### 1. Web Scraping y Crawling 
- Crawler: Navega automáticamente por sitios web descargando páginas
- Scraper: Extrae información específica de esas páginas
- Ejemplo: Recopilar tweets para analizar sentimientos sobre un producto

    ### 2. Etiquetado de datos
- Tomar datos sin etiquetar y agregarles el "objetivo"
- Ejemplo: Tienes 1000 fotos de perfil, pero necesitas marcar cuáles tienen personas
- Servicios como Amazon Mechanical Turk te ayudan con esto

    ### 3. APIs y JSON
- Usar interfaces oficiales de sitios web para obtener datos
- Ejemplo: Twitter API, Facebook API, etc.

#-------------------------Valor objetivo------------------------------
también llamado "target" o "etiqueta" es la respuesta correcta que quieres que tu modelo aprenda a predecir.

Con valores objetivo:
Foto 1: [imagen de gato] → Etiqueta: "gato"
Foto 2: [imagen de perro] → Etiqueta: "perro"
Foto 3: [imagen de gato] → Etiqueta: "gato"

Sin valores objetivo:
Foto 1: [imagen de gato] → ¿?
Foto 2: [imagen de perro] → ¿?
Foto 3: [imagen de gato] → ¿?

#------------------------Control de calidad de etiquetado------------------------------
La calidad de los datos después del etiquetado se puede mejorar utilizando los métodos para el control de calidad del etiquetado.

### voto mayoritario:
Por ejemplo, cada observación está etiquetada por tres evaluadores. La respuesta final es la elegida por la mayoría.

import pandas as pd
data = pd.read_csv('/datasets/heart_labeled.csv')
target = []
for i in range(data.shape[0]):
    labels = data.loc[i, ['label_1', 'label_2', 'label_3']]
    true_label = int(labels.mean() > 0.5)
    target.append(true_label)
data['target'] = target
print(data.head())

Étape 1 : labels.mean()
Quand tu as des valeurs 0 et 1, la moyenne te donne la proportion de 1 
Si labels = [0, 0, 1] → moyenne = (0+0+1)/3 = 0.33
Si labels = [0, 1, 1] → moyenne = (0+1+1)/3 = 0.67
Si labels = [1, 1, 1] → moyenne = (1+1+1)/3 = 1.0
Étape 2 : labels.mean() > 0.5

Cette comparaison retourne True ou False :
0.33 > 0.5 → False (minorité de 1)
0.67 > 0.5 → True (majorité de 1)
1.0 > 0.5 → True (majorité de 1)

Étape 3 : int(...)
Convertit le booléen en entier :
int(False) = 0
int(True) = 1

#-----------------------Fuga de objetivos------------------------------
ocurre cuando la información sobre el objetivo se filtra accidentalmente en las variables independientes.
Entonces, las características "Nivel del personaje" y "Total de horas dedicadas al juego" se conocen después de que el modelo hace predicciones. Para evitar la fuga de información, no utilices dichas funciones.
Son problemáticas porque dependen de si el jugador ya gastó dinero (que es exactamente lo que queremos predecir).

Ejemplo: Predecir si alguien tiene gripe
Objetivo: Crear un modelo que prediga si una persona tiene gripe
Datos que recopilamos:
SIN fuga de información:
- Edad de la persona
- Si tiene fiebre
- Si tiene dolor de cabeza
- Época del año

CON fuga de información:
- Medicamento que está tomando (¡Problema!)
- Días de baja médica (¡Problema!)

¿Por qué hay fuga?
El medicamento: Si alguien está tomando antigripal, ¡ya sabemos que tiene gripe! Es como hacer trampa en el examen.
Los días de baja: Solo se dan bajas médicas DESPUÉS de diagnosticar la gripe. Es información del futuro.

La regla de oro:
"¿Esta información existía ANTES de que quisiéramos hacer la predicción?"
Edad →  Sí existía
Medicamento → Solo existe DESPUÉS del diagnóstico

#-----------------------Validación cruzada------------------------------
Cada subconjunto representa la distribución del conjunto de datos original, pero con algunas desviaciones ( entrenamiento, prueba y validación)

Estas desviaciones son causadas por la aleatoriedad que ocurre al formar un subconjunto a través de una técnica de muestreo aleatorio utilizada en la función train_test_split. 

¿cómo podemos asegurarnos de que la distribución de cada subconjunto sea relativamente cercana al conjunto de datos original? 
¡La solución es tomar varias muestras al azar!

La validación cruzada(validación cruzada k-fold.) es una técnica que ayudará a entrenar y probar un modelo utilizando varias muestras formadas aleatoriamente. 
1. primero tenemos que decidir un valor k. Este valor indica el número de subconjuntos iguales en los que dividiremos nuestro conjunto de datos
2. dividir el conjunto de datos en k subconjuntos iguales
3. iteramos sobre estos subconjuntos y les asignamos diferentes propósitos.
    Por ejemplo, durante la primera iteración, el primer subconjunto se usará para la validación y los restantes son para entrenamiento.
    En la segunda iteración, el segundo subconjunto ahora se usa para la validación, mientras que los demas ahora se asignan con fines de entrenamiento.

tomando cada vez un nuevo subconjunto para la validación
En cada paso de la iteración, el modelo se entrena con los datos de entrenamiento y se evalúa con los datos de validación.

Cuando se completan todas las iteraciones de k, se realiza la evaluación final del modelo tomando la media de todas las puntuaciones de evaluación de k.
usa subconjuntos con contenido fijo que no cambia en cada etapa de entrenamiento y validación
 Cada observación pasa por el conjunto de entrenamiento y el conjunto de validación.

La validación cruzada es útil cuando necesitamos comparar modelos, seleccionar hiperparámetros o evaluar la utilidad de las características.
Minimiza la aleatoriedad de la división de datos y proporciona un resultado más preciso. 

El único inconveniente de la validación cruzada es el tiempo de cálculo, especialmente con muchas observaciones o un valor alto de k. Es mucho tiempo.

#Validación cruzada manual-Python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('/datasets/heart.csv')
features = data.drop(['target'], axis=1)
target = data['target']
scores = []
sample_size = int(len(data) / 3) #dividir el conjunto de datos en 3 subconjuntos
for i in range(0, len(data), sample_size):
    # Con 300 datos y sample_size=100:
    # Iteración 1: i = 0   (bloque del 0 al 99)
    # Iteración 2: i = 100 (bloque del 100 al 199)  
    # Iteración 3: i = 200 (bloque del 200 al 299)
    valid_indexes = list(range(i, i + sample_size))
    train_indexes = list(range(0, i)) + list(range(i + sample_size, len(data)))
    features_train = features.iloc[train_indexes]
    features_valid = features.iloc[valid_indexes]
    target_train = target.iloc[train_indexes]
    target_valid = target.iloc[valid_indexes]
    model = DecisionTreeClassifier(random_state=0)
    model = model.fit(features_train, target_train)
    score = model.score(features_valid, target_valid)
    scores.append(score)
final_score = sum(scores) / len(scores)
print('Valor de calidad promedio del modelo:', final_score)

#Validación cruzada sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
data = pd.read_csv('/datasets/heart.csv')
features = data.drop(['target'], axis=1)
target = data['target']
model = DecisionTreeClassifier(random_state=0)
score = cross_val_score(model, features, target, cv=5)
final_score = sum(score) / len(score)
print('Puntuación media de la evaluación del modelo:', final_score)
 
el modelo será entrenado en el proceso de validación cruzada, por lo que tenemos que pasarlo sin entrenar
La función devuelve una lista de valores de evaluación del modelo de cada validació (Cada valor es igual a model.score())
#La función cross_val_score() que usaste solo evalúa el modelo, pero no lo entrena para uso futuro.

Cada entrenamiento empieza desde cero: El modelo no "recuerda" nada del fold anterior
Se entrena un modelo nuevo cada vez: Con datos de entrenamiento diferentes
Solo importa el resultado final: El promedio de todos los scores

Comportamiento por defecto: OLVIDA todo
Cuando llamas .fit() múltiples veces, el modelo borra el conocimiento anterior:

¿Para qué sirve la validación cruzada?
No es para entrenar el modelo final - es para evaluar qué tan confiable es tu modelo antes de usarlo en el mundo real.

El flujo completo es:
Validación cruzada: "¿Qué tan bueno es este modelo?" → final_score = 0.85
Si el score es bueno: Entrenas el modelo final con TODOS los datos
Usas ese modelo final para predicciones reales

#-----------------------Que saber de la empresa donde aplico------------------------------
¿Cuál es el sector?
¿Cuál es la actividad principal de la empresa que aporta dinero?
¿La empresa trabaja con particulares u otras empresas?

Esto es lo que debes averiguar durante las primeras semanas:
Los objetivos clave de la empresa y su departamento para el año en curso y los periodos de informe
El lugar que ocupa tu departamento en la estructura de la empresa
¿Cuáles son los KPI para tu departamento y para ti personalmente?
¿A qué proceso del negocio perteneces?
¿Con quién estás trabajando? ¿Quiénes son tus colegas, clientes internos y contratistas?





























































































































































































































































































































































































































































































































































































































































































































































































































































































































