# Errores de sintaxis

Si el nombre de una columna consta de varias palabras, lo mejor es usar snake_case.

# Renombrar columnas

Para cambiar el nombre de las columnas, llama al método rename() con un diccionario como su argumento columns.

Las claves del diccionario deben ser los nombres anteriores
de las columnas, y los valores correspondientes deben ser los nuevos nombres.

    columns_new ={
        "Celestial bodies ": "celestial_bodies",
        "MIN": "min_distance",
        "MAX": "max_distance",
    }
    celestial = celestial.rename(columns = columns_new)
