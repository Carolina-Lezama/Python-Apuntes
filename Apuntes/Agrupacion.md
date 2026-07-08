# Agrupación de datos
La agrupación te permite dividir los datos en grupos según ciertos criterios.

## ¿Cuándo agrupar?
La agrupación se justifica cuando los datos caen lógicamente en grupos en función de una determinada característica y cuando los grupos son relevantes para la tarea en cuestión.

## Etapas de la agrupación
1. Dividir. Primero, divide los datos en grupos según un criterio determinado.
2. Aplicar. A continuación, aplica métodos de cálculo a cada grupo.
3. Combinar. Finalmente, los resultados son almacenados en una nueva estructura de datos.

## Agrupación en pandas
En pandas, agrupamos los datos utilizando el método groupby(), que hace lo siguiente:

- Toma el nombre de una columna en la que se deben agrupar los datos como
argumento. Este parámetro se llama by=.

- Devuelve un objeto de un tipo especial: DataFrameGroupBy. Son datos agrupados. Si les aplicas un método de pandas, se convertirán en una nueva estructura de datos.

    print(df_exo1.groupby(by='discovered').count())