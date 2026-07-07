# La desviacion estandar

Una medida que nos dice qué tan "rebeldes" o "dispersos" son los datos.

Nos indica qué tanto se alejan, en promedio, los datos respecto al valor central (el promedio o la media).

### Imagina que tenemos dos grupos de estudiantes y ambos grupos tienen un promedio de calificación de 80.

Grupo A: Todos sacaron exactamente 80. (Aquí no hay dispersión; la desviación estándar es 0).

Grupo B: Unos sacaron 100, otros 60, otros 90 y otros 70. El promedio sigue siendo 80, pero los datos están muy variados. (Aquí hay una desviación estándar alta).

Un valor por sí solo no significa si es "mucho" o "poco". Todo depende de cuál sea tu promedio y de qué estés midiendo.

Significa que, en promedio, los datos se alejan x unidades de la media.

# Descripciones numéricas

Puedes llamar a describe() en un DataFrame o en un Series

        print(data.describe())
        print(data['capacity_mw'].describe())

![alt text](imagenes/image-3.png)

De forma predeterminada, se ignoran las columnas no numericas

# Descripciones no numéricas

        print(data['country'].describe())

![alt text](imagenes/image-5.png)

'top': el valor que aparece con más frecuencia;

Solo las columnas no numéricas:

        print(data.describe(include='object'))

# Descripcion para todos los datos

        print(data.describe(include='all'))

![alt text](imagenes/image-5.png)

# Error Cuadrático Medio (ECM)

MSE (Mean Squared Error).

RMSE (Root Mean Squared Error).

Medida estadística que se utiliza para evaluar la calidad de un modelo
predictivo. Nos indica qué tan lejos están, en promedio, las
predicciones de nuestro modelo de los valores reales.

## ¿Cómo se calcula el ECM?

1. Calcular la diferencia entre el valor real y el valor predicho para cada dato
2. Elevar al cuadrado cada diferencia: penalizar más los errores
   grandes
3. Calcular el promedio

Aunque el ECM no tiene una interpretación directa en unidades
físicas, nos proporciona una medida numérica de la precisión de nuestro modelo
