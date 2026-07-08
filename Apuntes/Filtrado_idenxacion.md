# Indexación lógica (booleana)
Encontrar todas las filas en las que se cumple cierta condición.

        print(df['genre'] == 'pop')
        print(df.loc[:, 'genre'] == 'pop')

El boleano indica si se cumplio la condicion o no. 

Ahora podemos utilizar el resultado obtenido en el paso anterior para filtrar la tabla
original.

        print(df.loc[df.loc[:, 'genre'] == 'pop'])
        print(df.loc[df['genre'] == 'pop'])
        
        print(df[df['genre'] == 'pop'])

# Indexación por coordenadas
La indexación permite acceder a una celda determinada de la tabla utilizando dos coordenadas: el número de la fila y el nombre de la columna.

Mediante la indexación es posible solicitar celdas individuales y grupos de celdas. Por ejemplo, puedes acceder a:
- todas las celdas de una fila determinada;
- todas las celdas de varias filas;
- todas las celdas de un rango de filas.

        result = df.loc[4, 'genre']

De forma similar a la segmentación de listas, puedes obtener un rango de valores de
una tabla especificando el principio y el final de una segmentación, separados por dos
puntos :

![alt text](image-2.png)