# Valores ausentes

Valores ausentes, indicados con NaN, que significa "not a number" y es una forma común de marcar valores ausentes, cuando una celda no está llena por alguna razón.

El método isna() si se encuentra un valor ausente, devuelve True; si
no, devuelve False.

    print(colera['country'] .isna().sum())
    print(colera.isna().sum())

O tambien:

    print(colera.isnull().sum())

# Formas de procesar valores ausentes

Las filas de una tabla pueden eliminarse por completo si han perdido su valor debido a los valores ausentes.

A veces, los valores ausentes se remplazan con otros valores. Esto se puede hacer cuando los valores ausentes no son muy relevantes para nuestro análisis, pero las filas o columnas aún contienen datos valiosos.

    colera['imported_cases'] = colera['imported_cases'].fillna(0)

# Eliminar filas dropna().

Este método elimina las filas con al menos un valor ausente. También puedes
especificar una lista de columnas en su parámetro subset= para que elimine filas con valores nulos solo en esas columnas.

Así es como funciona (dropna revisa en las columnas indicadas en subset si contiene valores nulos en caso afirmativo dropna elimina la fila)

    colera = colera.dropna(subset=['total_cases', 'deaths', 'case_fatality_rate'])

axis= este argumento nos permite especificar si queremos eliminar filas o columnas. Si pasamos el string 'columns' a axis=, eliminará las columnas que tengan valores ausentes.

    colera = colera.dropna(axis='columns')

# Eliminar columnas controladamente

El método drop() para controlar qué columnas quieres eliminar.

    colera = colera.drop(labels=['notes'], axis='columns')
