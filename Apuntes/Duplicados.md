# Valores duplicados

Valores duplicados, cuando dos o más filas son exactamente iguales.

- Llamamos al método drop_duplicates() y obtenemos un DataFrame sin duplicados,
  pero con índices rotos.
- Luego usamos el método reset_index() para restablecer los índices de las filas y eliminar la columna 'index'.+

# Eliminación de duplicados implícitos

Utiliza el método replace() para corregir la ortografía incorrecta o alternativa.

    tennis['name'] = tennis['name'].replace('Roger Federerr', 'Roger Federer')
