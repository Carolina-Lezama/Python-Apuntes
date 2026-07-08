# ¿Qué es la regresión lineal?

Es una técnica de análisis de datos que predice el valor de datos desconocidos mediante el uso de otro valor de datos relacionado y conocido(predecir el valor de una variable en función del valor de otra variable).

- La variable que deseas predecir se llama variable dependiente(y).
- La variable que está utilizando para predecir el valor de otra se llama variable independiente(x).

Los científicos de datos primero entrenan el algoritmo con datos conocidos (etiquetados o no) y, luego, utilizan el algoritmo para predecir valores desconocidos.

Debe existir una relación x y y. Para determinar esta relación, los científicos de datos crean una gráfica de dispersión.

Los científicos de datos utilizan residuos para medir la precisión de la predicción. Un residuo es la diferencia entre los datos observados y el valor previsto.

### Normalidad

Las gráficas Q-Q, determinan si los residuos se distribuyen normalmente. Si los residuos no están normalizados, puede probar los datos para detectar valores atípicos o aleatorios. Eliminar los valores atípicos o realizar transformaciones no lineales puede solucionar el problema.

### Homocedasticidad

La homocedasticidad supone que los residuos tienen una variación constante o desviación estándar de la media para cada valor de x. De lo contrario, es posible que los resultados del análisis no sean precisos. Si no se cumple esta suposición, es posible que tenga que cambiar la variable dependiente.

# ¿Cuáles son los tipos de regresión lineal?

- Regresión lineal simple: Eelación entre dos variables

- Regresión lineal múltiple: Una variable
dependiente y múltiples variables independientes

# Regresión logística

Medir la probabilidad de que se produzca un evento. La predicción es un valor entre 0 y 1, donde 0 indica un evento que es poco probable que ocurra y 1 indica una probabilidad máxima de que suceda.

Análisis de clasificación utilizado para predecir el resultado de una variable categórica (número limitado de categorías). Es útil para modelar la probabilidad de un evento ocurriendo en función de otros factores.

La regresión logística es una técnica de análisis de datos que utiliza las matemáticas para encontrar las relaciones entre dos factores de datos. Luego, utiliza esta relación para predecir el valor de uno de esos factores basándose en el otro.

Para la regresión logística, hay que formular la pregunta para obtener resultados concretos

### Regresión logística binaria

Funciona bien para problemas de clasificación binaria que solo tienen dos resultados posibles. 

### Regresión logística multinomial

La regresión multinomial puede analizar problemas que tienen varios resultados posibles, siempre y cuando el número de resultados sea finito. Por ejemplo, puede predecir si los precios de la vivienda aumentarán un 25 %, 50 %, 75 % o 100 % en función de los datos de población, pero no puede predecir el valor exacto de una casa. La regresión logística multinomial funciona mapeando los valores de resultado con diferentes valores entre 0 y 1.

### Regresión logística ordinal

Es un tipo especial de regresión multinomial para problemas en los que los números representan rangos en lugar de valores reales.

## Regresión logística vs. regresión lineal

La regresión lineal predice una variable dependiente continua mediante el uso de un conjunto dado de variables independientes. Una variable continua puede tener un rango de valores.

A diferencia de la regresión lineal, la regresión logística es un algoritmo de clasificación. No puede predecir los valores reales de los datos continuos.

La regresión logística es menos compleja y requiere menos recursos informáticos que el aprendizaje profundo. Y lo que es más importante, los desarrolladores no pueden investigar ni modificar los cálculos de aprendizaje profundo, debido a su naturaleza compleja y dirigida por la máquina. Por otro lado, los cálculos de regresión logística son transparentes y más fáciles de solucionar.

La regresión logística binaria es el algoritmo de clasificación por excelencia en Machine Learning.

Existen algoritmos de clasificación (e incluso versiones de la regresión logística) que soportan más de dos categorías. A esto se le conoce en la industria como Clasificación Multiclase (Multiclass Classification).

