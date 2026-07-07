# Pasos para la creación del Modelo Regresión Lineal.

1. Obtener informacion
2. Realizar limpieza
3. Definir x y y
4. Hacer el split de entrenamiento, test y valid
5. One-Hot Encoding.
6. Crear el modelo LinearRegression()
7. Entrenar el modelo con .fit(x, y)
8. Usar predict para obtener los valores a comprobar
9. Evaluar el modelo

Hacer el One-Hot Encoding (o cualquier tipo de preprocesamiento y transformación de datos) antes de separar tus datos es uno de los errores más comunes, y la razón es la Fuga de datos.

## Dimenciones de X y Y

Por defecto, y es una Serie de Pandas, básicamente una lista unidimensional (un vector plano).

El problema es que LinearRegression de sklearn es muy estricto: siempre espera una matriz de dos dimensiones (filas, columnas) tanto para las características (X) como para las etiquetas (Y)

#### ¿Qué hace reshape((-1, 1))?

El método .reshape() cambia la "forma" de tu arreglo sin cambiar los datos que contiene.

La sintaxis recibe dos argumentos: (filas, columnas). Al pasarle (-1, 1), le estás dando una instrucción muy específica a Python:

El 1 (Columnas)
El -1 (Filas): Todas las filas en una sola columna
