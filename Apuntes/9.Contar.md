# Contar valores ausentes value_counts()

En vez de sumar los valores obtenidos con isna(), podemos contar los valores ausentes con el método value_counts(). 

Al llamarlo en una sola columna (es decir, un Series), devuelve la cantidad de veces que cada valor único aparece en esa columna.

Este método tiene un parámetro llamado dropna=, que se establece por defecto en True. Esto significa que value_counts() excluirá los valores None o NaN

La salida se ordena en orden descendente según el recuento de cada valor. Alternativamente,
podemos ordenar la salida alfabéticamente según los nombres de los valores. Para hacerlo,
podemos utilizar el método sort_index().
