#------------IMPORTAR PANDAS----------------
import pandas as pd

#------------ABRIR ARCHIVOS CSV----------------
#Se guardan en una variable
df = pd.read_csv('/datasets/music.csv', sep=',') 
# Por defecto: sep=',' 
# Otros valores: sep='\\t'(tabulaciones) sep='|' sep=';'
# Forma en que estan separados los datos

#.............Renombrar columnas de un dataframe.............
column_names = ['user_id', 'genre', 'Artist', 'track', 'total play']
df = pd.read_csv('/datasets/music.csv', header=None , names=column_names) 
# Por si no hay cabeceras, evitar tomar filas como cabecera.
# header=None Indica a read_csv que la primera fila no contiene los nombres de las columnas, sino la primera fila de datos reales

#.............Volver a flotantes.............
df = pd.read_csv('/datasets/music.csv', decimal=',') 
# Volver a los numeros 9,1 a 9.1

#------------ABRIR ARCHIVOS EXCEL---------------
# Solo trae la primera hoja por defecto
df = pd.read_excel('/datasets/online.xlsx') 

dfs = pd.read_excel('/datasets/product.xlsx', sheet_name=None)
# Cargar todas las hojas

df = pd.read_excel('/datasets/online.xlsx', sheet_name='hoja_2') 
# Cargar una hoja por nombre ESPECIFICA

df = pd.read_excel('/datasets/online.xlsx', sheet_name=1) 
# Cargar hoja por indice

dfs = pd.read_excel('/datasets/product.xlsx', sheet_name=['product_categories', 'reviews']) 
# Cargar lista de hojas por nombre

df_dict  = pd.read_excel('/datasets/product.xlsx', sheet_name=[0, 2, 3]) 
# Cargar lista de hojas por indice, Hojas exactas

df_dict = pd.read_excel('/datasets/product.xlsx', sheet_name=list(range(0, 2)))
#  Leer un rango continuo

#------------ GUARDAR RUTAS ---------------
file_path = '/datasets/diabetes.csv'
df = pd.read_csv(file_path)

#------------ RELLENAR VALORES CATEGORICOS AUSENTES----------------
df = pd.read_csv('/datasets/music_log_chpt_11.csv', keep_default_na=False) #vuelve cadenas vacias en vez de NaN

#------------FUNCION LOC----------------
#Existe la notación abreviada y la completa

#.............Notacion completa.............
# .loc[indice(s) , columna(s)]
# El índice (filas) siempre se especifica primero y las columnas son opcionales.
# No puedes acceder a una columna directamente haciendo df.loc['columna']
#  Si no pones el segundo argumento, Pandas asume de forma automática que quieres todas las columnas para las filas seleccionadas.

df.loc[4, 'genre'] # Una celda - una columna
df.loc[:, 'genre'] # Todas las filas - una columna  
df.loc[:, ['genre', 'Artist']] # Todas las filas - Varias columnas no consecutivas
df.loc[:,'total play': 'genre'] # Todas las filas - Múltiples columnas consecutivas (slice)
df.loc[5, 'total play': 'genre'] # Una fila - varias columnas consecutivas
df.loc[1] # Una fila completa por indice

# Slice de filas
df.loc[1:] # Desde la fila 1 hasta el final 
df.loc[:3] # Desde el inicio hasta la fila 3 
df.loc[2:5] # Desde la fila 2 hasta la fila 5 

# Los indices no necesariamente son numericos
df.loc['state 1': 'state 3', 'flower': 'insect'] # Slice de indices, slice de columnas
df.loc['state 1': 'state 3', 'flower'] # Slice de indices, de una columna
df.loc[[0, 1], ['col_name_1', 'col_name_2']] # Devuelve las filas 0 y 1, y las columnas especificadas (NO rango)

#.............Notacion abreviada.............
# Busca columnas primero, y las filas son opcionales.
# Para obtener un slice (corte) de columnas con la notación abreviada, no puedes hacerlo directamente 

df['genre'] # Columna(s)
df[['genre', 'Artist']] # Columnas no consecutivas

df[1:] #fila(s)
df[:3] 
df[2:5]

# Notación abreviada (Usando .columns)
df[df.columns[2:5]]  # Corta las columnas de la posición 2 a la 4

#.............¿Cual elegir?.............
# df.loc[...] - Piensa primero en Filas
# df[...] - Piensa primero en Columnas.

#------------CREAR UN DATAFRAME----------------
sales_data = [
    ['Laptop' , 'North America', 120 , 120000], # Cada valor es una celda horizontalmente
    ['Smartphone' , 'Europe' , 340 ,170000],
    ['Tablet' , 'Asia' , 210 , 63000],
    ['Headphones' , 'South America' , 150 , 45000],
    ['Smartwatch' , 'Africa' , 95 , 28500]
]
values = ['object', 'capital', "price", "cantite"]
sales = pd.DataFrame(data=sales_data, columns=values) # Lista de datos, nombres de las columnas

atlas = [
['France', 'Paris'],
['Russia', 'Moscow'],
['China', 'Beijing'],
['Mexico', 'Mexico City'],
['Egypt', 'Cairo']
]
geography = ['country', 'capital']
resultado = pd.DataFrame(data = atlas, columns = geography)

states  = ['Alabama', 'Alaska', 'Arizona', 'Arkansas']
flowers = ['Camellia', 'Forget-me-not', 'Saguaro cactus blossom', 'Apple blossom']
insects = ['Monarch butterfly', 'Four-spotted skimmer dragonfly', 'Two-tailed swallowtail', 'European honey bee']
index   = ['state 1', 'state 2', 'state 3', 'state 4']
df = pd.DataFrame({'state': states, 'flower': flowers, 'insect': insects}, index=index) #crear columnas con listas y asignar indices personalizados
# En forma de diccionario

measurements = [['Sun', 146, 152],
                ['Moon', 0.36, 0.41],
                ['Mercury', 82, 217],
                ['Venus', 38, 261],
                ['Mars', 56, 401],
                ['Jupiter', 588, 968],
                ['Saturn', 1195, 1660],
                ['Uranus', 2750, 3150],
                ['Neptune', 4300, 4700],
                ['Halley\'s comet', 6, 5400]]
header = ['Celestial bodies ','MIN', 'MAX']
celestial = pd.DataFrame(data=measurements, columns=header)

#------------FUNCIONES EN PANDAS BASE----------------
print(df.head(2)) # Primeras 5 filas por defecto
print(df.tail(6)) # Ultimas 5 filas por defecto
print(df.sample(5)) # Muestra aleatoria, pasamos como argumento la cantidad, por defecto 1.

print(df.dtypes) # Columnas y tipos de datos
print(df.shape) # Numero de filas y columnas
print(df.columns) # Nombres de las columnas

todo = df.info() # Informacion general del DataFrame

#------------FUNCIONES DE AGREGACION----------------
# Por defecto y si se aplican a todo el DataFrame completo, devuelven una sola "fila" (una Serie/un escalar)
# Devuelve una Serie cuando la aplicas a un DataFrame completo (o a varias columnas a la vez), no a una sola columna

# Si las usas con la función groupby(), entonces sí devolverán múltiples filas.

#.............Suma.............
duracion = df[df['user_id'] == '123456']['total play'].sum() # Devuelve escalar
duracion = df[df['user_id'] == '123456'][['total play', 'total_pauses']].sum() # Devuelve escalar
# 1. Filtramos por columna y valor, regresa df filtrado (filas completas que cumplen)
# 2. Se selecciona una fila especifica del df filtrado y se aplica la funcion

#.............Promedio.............
sin_filtro = df['total play'].mean() 
filtrado = df[df['genre'] == 'Pop']['total play'].mean()

#.............Contar.............
contar = df[df['total play'] > 30]['total play'].count()
# Contar cuántas filas tienen un total play mayor a 30
contar = (df['total play'] > 30).sum() # Anterior hecho con sumar booleanos

#------------INDEXACION LOGICA O BOOLEANA----------------
# Aislar datos a cierto tipo, solo tendra las filas que cumplan la condicion

# Notacion mezclada:
df.loc[df['genre'] == 'Rock'] # Completa usando abreviada dentro

#Notacion completa:
df.loc[df.loc[:,'genre'] == 'Rock']

# Notacion abreviada:
resultado = df[df['genre'] == 'Rock'] # Abreviada /abreviada
resultado = df[df['genre'] == 'Pop'] 
resultado = df[df['total play'] >= 90] 
resultado = df[df['total play'] <= 90] 
resultado = df[df['device_type'] != 'desktop']
resultado =  df[df['device_type'] == 'mobile']

#.............Mas de 2 condiciones de filtrado.............
# Condicional doble - separada
ventas_filtradas = df[df['sucursal'] == 'Centro'] 
ventas_filtradas = ventas_filtradas[ventas_filtradas['precio_unitario'] > 25]

# Condicional doble - unidas
# True and True, ambas deben cumplirse
ventas_filtradas = df[(df['sucursal'] == 'Centro') & (df['precio_unitario'] > 25)] #condicional doble, unida

#------------ FUNCION RENAME----------------
# Diccionario con: nombre anterior : nuevo nombre
new_columns ={ 
    '  user_id': 'user_id', 
    'total play': 'total_play',
    'Artist': 'artist'
    }
df = df.rename(columns = new_columns)
df.rename(columns=new_columns, inplace=True)
# inplace=True, sirve para modificar el DataFrame original directamente, en lugar de crear y devolver una copia nueva con los cambios

df.rename(columns={'  user_id': 'user_id', 'total play': 'total_play', 'Artist': 'artist' }, inplace=True)

df = df.rename(columns={'userid': 'user_id'})

#.............Ciclo para renombrar columnas.............
# 1. Construir una lista
# 2. Crear ciclo que itera sobre las columnas
# 3. Normalizacion
# 4. Renombrar

columna_nombre = [] 
for viejo_nombre in df.columns: 
    nuevo_nombre = viejo_nombre.strip().lower().replace(" ","_")
    columna_nombre.append(nuevo_nombre)
df.columns = columna_nombre

new_col_names = []
for old_name in celestial.columns:
    # Elimina los espacios al principio y al final
    name_stripped = old_name.strip()
    
    # De mayusculas a minusculas
    name_lowered = name_stripped.lower()
    
    # Reemplazar espacios
    name_no_spaces = name_lowered.replace(' ', '_')
    
    new_col_names.append(name_no_spaces)
celestial.columns = new_col_names

#------------DESCRIBE---------------
# Resumen estadístico detallado para cada una de las columnas; por defecto solo numericas
print(df.describe()) 
print(df.describe(include='object')) 
print(df.describe(include='all')) # Todas las columnas

#------------ CONTAR VALORES AUSENTES ----------------
df.isna().sum() # Si es true es ausente, al final se suman los ausentes.

#------------ LLENAR VALORES AUSENTES ----------------
# Para TODO el DataFrame
df.fillna(0, inplace=True)

# Para múltiples columnas
df[['precio', 'cantidad']] = df[['precio', 'cantidad']].fillna(0)

# Para varias columnas con DIFERENTES valores
valores_de_relleno = {
    'edad': 25,                
    'ciudad': 'Desconocida',  
    'suscripcion': False     
}
df.fillna(valores_de_relleno, inplace=True)

# Para una columna
colera['imported_cases'] = colera['imported_cases'].fillna(0)
colera['imported_cases'].fillna(0, inplace=True) 
df['genre'].fillna(0, inplace = True)
df['email'] = df['email'].fillna(value='')
df['col'].fillna('no_info', inplace=True)

# Para varias columnas con ciclo
columns_to_replace = ['imported_cases']
for col in columns_to_replace:
    colera[col].fillna(0, inplace=True)

#------------ ELIMINAR VALORES AUSENTES ----------------
# El comportamiento por defecto (Borrar fila si hay CUALQUIER nulo)
df.dropna(inplace=True)

# Buscar nulos solo en columnas específicas (subset)
# Elimina si falta algun dato dentro de las columnas listadas
df.dropna(subset=['precio', 'cantidad'], inplace=True)
df.dropna(subset=['genre','total play'], inplace=True) 
colera = colera.dropna(subset=['total_cases', 'deaths', 'case_fatality_rate'])

# Borrar columnas enteras en lugar de filas (axis=1)
df = df.dropna(axis = 'columns') #eliminar la columna si tiene al menos un ausente
colera = colera.dropna(axis='columns')
df.dropna(axis=1, inplace=True)

# Borrar filas completas
df = df.dropna(axis = 'rows') #eliminar la fila si tiene al menos un ausente
df.dropna(axis=0, inplace=True)

# Borrar solo cuando TODO está vacío (how='all')
# Solo elimina la fila si TODAS sus columnas tienen NaN
df.dropna(how='all', inplace=True)

# Salvar la fila si tiene un mínimo de datos válidos (thresh)
# La fila necesita tener al menos 3 datos válidos (no nulos) para salvarse.
df.dropna(thresh=3, inplace=True)

# Para columna
df['my column'] = df['my column'].dropna()

#------------ ELIMINAR FILAS O COLUMNAS QUE YO INDIQUE----------------
# Eliminar una o mas columna especifica
colera = colera.drop(labels=['notes'], axis='columns')
df = df.drop(labels=['notes', 'puede_ser_una_o_varias'], axis='columns') 

colera.drop(labels=['notes'], axis='columns', inplace=True)

df = df.drop(labels=0, axis='columns') #eliminar una columna por indice

#------------ VALORES DUPLICADOS ----------------
df.duplicated().sum() 
# Devuelve boolean si hay duplicados, se suman

df[df.duplicated(keep=False)] # Devuelve la posiciones de los duplicados incluida la primera aparicion

#------------ ELIMINAR FILAS COMPLETAMENTE DUPLICADAS ----------------
df = df.drop_duplicates().reset_index(drop=True) 
# Elimina duplicados, resetear los indices, borrar anterior lista de indices
df.drop_duplicates(inplace=True) 

df.drop_duplicates(subset='col_1') #Busca y elimina duplicados solo considerando la columna
df.drop_duplicates(subset=['col1', 'col2']) #Busca y elimina duplicados basandose en varias columnas

#------------UNIQUE Y NUNIQUE---------------
# Solo una columna a la vez
df['name'].unique() #valores unicos en una columna, sirve para verificar duplicados por mala escritura, etc. devuelve array
df['name'].nunique() #da el numero de valores unicos, devuelve entero

#.............Automatización con funciones personalizadas.............
duplicates = ['Roger Federerr', 'Roger Fedrer'] # Una lista de nombres mal escritos
name = 'Roger Federer' # El nombre correcto

# 1. Toma argumentos
# 2. Bucle que itera en los valores errones
# 3. Reemplaza valores_de_relleno
# 3. Regresa el dataframe

def replace_wrong_values(df, column, wrong_values, correct_value):
    for wrong_value in wrong_values:
        df[column] = df[column].replace(wrong_values, correct_value)
    return df

tennis = replace_wrong_values(df, 'name', duplicates, name) # Llamar a la función

#------------ AGRUPAMIENTO DE DATOS ----------------
# Agrupar por columna y aplicar una funcion de agregacion a cada grupo formado
print(df.groupby(by='discovered').count())
print(df.groupby('discovered').count())

# 1. Argumento de agrupamiento por el que se dividira
# 2. Columna a las que se le aplicara la funcion 

# [] Actua como un filtro, en qué columna específica quieres aplicar la operación matemática
exo_number = df.groupby('discovered')['radius'].count()
exo_radius_sum = df.groupby('discovered')['radius'].sum()
grouped_stage_count = df.groupby('Stage')['Digimon'].count()
grouped_stage_sum = df.groupby('Stage')['Lv 50 HP'].sum()
grouped_stage_mean = df.groupby('Stage')['Lv50 Spd'].mean()

mean_score = df.groupby('genre')['critic_score'].mean()
    # df es un DataFrame
    # Stage es una columna de df, se agrupara por esta
    # Digimon es otra columna de df, a la que se le aplicara la funcion count() para cada grupo creado por los valores de Stage
    # .count() cuenta los valores no nulos dentro de cada grupo

# Cambia el índice de fila de los datos a las claves por las que estamos agrupando.

groupby_data = df.groupby(['genre', 'platform'])['eu_sales'].mean()
print(groupby_data)

grp = df.groupby(['platform', 'genre']) #agrupar por varios campos 
print(grp['critic_score'].mean())

#------------ METODOS PARA STRINGS DE FILAS----------------
# Minuculas
# Misma columnas 
df['col_1'] = df['col_1'].str.lower() # Convierte a minusculas los strings de una columna
# Diferente columna
df['item_lowercase'] = df['item'].str.lower() 

# Reemplazar
df['item_modified'] = df['item'].str.replace('GB', 'gb') 
df['genre'] = df['genre'].replace('ranchera','rancheras') 
df['source'] = df['source'].replace('','email')

# Primera Letra De Cada Palabra
df['city'] = df['city'].str.strip().str.title() 

#------------ FILTRAR CON NaN----------------
# Filtrar y quedarte únicamente con las filas donde la columna NO está vacía (elimina los valores nulos o NaN)
df[~df['source'].isna()] 

#------------ CONTAR VALORES----------------
# Toma una columna y cuenta cuántas veces aparece cada valor único, ordenando los resultados de mayor a menor frecuencia.

# Obtener porcentajes - cada conteo entre el total de filas
df['order_status'].value_counts(normalize=True)
# delivered: 0.97 (97%)
# canceled:  0.01 (1%)

# Hacer visibles los datos faltantes
# Este parámetro obligará a la función a contar los NaN como si fueran una categoría más.
df['col_1'].value_counts(dropna=False) 

# Agrupar números continuos en rangos
# Agrupará todas las en grandes bloques o rangos, mostrándo cuantas categorias caen dentro de cada rango.
df['col_1'].value_counts(bins=5)

# Orden de mostrar los resultados
df['col_1'].value_counts(ascending=True) 

# Contar valores, ordenados por indice(años) y no por el valor contado
# Por defecto, value_counts() ordena los resultados de mayor a menor frecuencia
df_year_of_release = df['year_of_release'].value_counts().sort_index() 

#------------OBJETO SERIES Y DETERMINAR EL INDEX---------------
# Pandas enumerará las filas automáticamente desde el 0 en adelante

# Usar string como index
oceans = pd.Series(
    ['Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic'],
    index=['A', 'B', 'C', 'D', 'E']
                ) 
# Renombrar los index a int
oceans.index = [1, 2, 3, 4, 5] 

#------------ ORDENAR DATOS ----------------
# .sort_values() es el equivalente directo y exacto de ORDER BY en SQL.
# Por valor numerico o string

# Ordena toda la tabla dandole una(s) columna(s) por la cual ordenar
df.sort_values(by='Stage', ascending=True, inplace=True) # ASC por defecto
df.sort_values(by=['categoria', 'precio'], ascending=[True, False])

# Ordenar una sola columna
df['radius'].sort_values(inplace=True) 

#.............Ordenar el index.............
# ordena usando las etiquetas de las filas (el índice)

# Ordenado por años
df_year_of_release = df['year_of_release'].value_counts().sort_index() 
# Ordern por años invertido (De más reciente a más antiguo)
df_year_of_release = df['year_of_release'].value_counts().sort_index(ascending=False)

# Ordenado por pais alfabeticamente
ventas_por_pais = df['pais'].value_counts().sort_index()

# Sin contar valores, solo acomodar index 
# Acomoda todas las filas del DataFrame basándose en su número de índice original
df.sort_index(inplace=True)

# Filtrado y ordenamiento
metal_ordenado = df[df['genre'] == 'metal'].sort_values(by='total_play', ascending=False) 

# Doble filtrado y ordenamiento
exo_small_14 = df[df['radius'] < 1] 
exo_small_14 = exo_small_14[exo_small_14['discovered'] == 2014]
exo_small_14 = exo_small_14.sort_values(by='radius', ascending=False)

#------------INDEXACION ILOC[]----------------
# .loc busca por nombres o etiquetas explícitas
# .iloc busca por posición numérica entera (la coordenada matemática exacta, empezando desde cero)

# Sintaxis: df.iloc[posicion_fila, posicion_columna]
# Se puede usar indexacion negativa

# Fila - Columa
df.iloc[4, 1]
df.iloc[3, 2]

# Slice de filas
df.iloc[0:3]

# Slice de columna
df.iloc[[0, 2], 1:]
df[df['publisher'] == 'other'].iloc[:,1:5]

#------------CAMBIAR INDICES CON SET_INDEX() PARA DATAFRAME----------------
# Toma una columna existente del df 
# Reemplaza los con los valores de esa columna
# #sa columna deja de exisitir en otros lados

df = df.set_index('state') 
df.index.name = None 
# Borrar el nombre de la columna, no tendremos titulo en los index.

#------------FILTRADO CON EL METODO query()----------------
# Requiere un string (representa la consulta)
# Los corchetes dobles al final actúan como un filtro de selección de columnas.

# Filtros simples (Números y Textos)
# MODO TRADICIONAL:
df_filtrado = pedidos[pedidos['precio'] > 150.00]

# MODO QUERY:
df_filtrado = pedidos.query("precio > 150.00")

pedidos_belleza = pedidos.query("categoria == 'belleza_salud'")

df.query("publisher == 'Nintendo'")['name'].head()

df.query("jp_sales > 1")[['name', 'jp_sales']]

cols = ['name', 'publisher', 'developer']
df_filtered = df.query("publisher == developer")[cols]

df.query("(platform not in @handhelds)")[['name', 'platform']]

df.query("genre not in @s_genres")[oceans]

df.query('year_of_release in [2006, 2007, 2008]')


# Múltiples condiciones (AND / OR)
# MODO TRADICIONAL:
alto_valor = pedidos[(pedidos['precio'] > 100) & (pedidos['estado_cliente'] == 'SP')]

# MODO QUERY:
alto_valor = pedidos.query("precio > 100 and estado_cliente == 'SP'")

df.query('year_of_release >= 2006 and year_of_release <= 2008')

df.query("platform == 'Wii' and genre != 'Sports'").head()

q_string = "na_sales >= 1 or eu_sales >= 1 or jp_sales >= 1"
print(df.query(q_string).head())

cols = ['name', 'developer', 'na_sales', 'eu_sales', 'jp_sales']
q_string = "na_sales > 0 and eu_sales > 0 and jp_sales > 0"
df_filtered = df.query(q_string)[cols]


# Variables externas (El símbolo @)
umbral_precio_optimo = 250.50
ventas_premium = pedidos.query("precio >= @umbral_precio_optimo") # Inyectar la variable dentro del query

handhelds = ['PC', 'Phone', 'TV']
df.query("~(platform in @handhelds)")[['name', 'platform']]

our_list = [2, 5, 10]
df = pd.DataFrame(
    {
        'a': [2, 3, 10, 11, 12],
        'b': [5, 4, 3, 2, 1],
        'c': ['X', 'Y', 'Y', 'Y', 'Z'],
    }
)
print(df.query("a in @our_list")) # Todas las filas de df donde los valores de 'a' están presentes en our_list

our_df = pd.DataFrame(index=[10, 11, 12]) # Usando indices de otro datadrame como filtro
print(df.query("a in @our_df.index"))

our_series = pd.Series([10, 11, 12])
print(df.query("a in @our_series")) # .index para probar con indices del series

our_dict = {0: 10, 3: 11, 12: 12}
print(df.query("a in @our_dict.values()")) 

# Usando el nombre de una columna, se usa . en lugar de['']
our_df = pd.DataFrame({'a1': [10, 11, 12]})
print(df.query("a in @our_df.a1"))

developers = ['SquareSoft', 'Enix Corporation', 'Square Enix']
cols = ['name', 'developer', 'na_sales', 'eu_sales', 'jp_sales']
q_string = "jp_sales > (na_sales + eu_sales) and jp_sales > 0 and na_sales > 0 and eu_sales > 0 and developer in @developers"
df_filtered = df.query(q_string)[cols]


# ¿Qué pasa si mis columnas tienen espacios?
# Envolver el nombre de esa columna en backticks (comillas invertidas `):
pedidos.query("`valor del flete` < 20.00")

#------------FILTRADO CON EL METODO isin()----------------

# Comprueba si valores en una lista coinciden con alguno de los valores en una columna del df
# Adecuado cuando tenemos muchas condiciones que comprobar
# Se puede usar ~  (Inversion de valor booleano)
# .isin() devuelve una mascara booleana

handhelds = ['3DS', 'DS', 'GB', 'GBA', 'PSP'] # Lista de referencia
df[df['platform'].isin(handhelds)][['name', 'platform']] 
# Comprueba si los valores de la columna 'platform' son iguales a algún valor de la lista handhelds

s_genres = ['M']
df_filtered = df[~df['genre'].isin(s_genres)][cols]

productos_más_vendidos = ['toalla', 'perfume', 'manzana']
ventas_filtradas = df[df['producto'].isin(productos_más_vendidos)]









resultado = df[((df['critic_score'] >= 90) | (df['user_score'] >= 9)) & (df['genre'].isin(['Role-Playing', 'Strategy', 'Puzzle']))
df[df['year_of_release'].isin([2006, 2007, 2008])]]

#------------ FILTRANDO POR CONDICIONALES ----------------
df_filtered = df[(df['year_of_release'] >= 1980) & (df['year_of_release'] < 1990)]
df[(df['year_of_release'] >= 2006) & (df['year_of_release'] <= 2008)]

large_cities = cities[cities['population'] > 100000]['city_id']
clients_in_large_cities = clients[clients['city_id'].isin(large_cities)]

df.loc[df['name'] == 'Tetris', 'year_released'] = 1984

#------------ METODO WHERE() ----------------
# si es true se mantiene, si es false se cambia
data = pd.Series([10, 20, 30, 40, 50])
filtered = data.where(data <= 30, 0) # Reemplazar todos los valores mayores a 30 por 0
df['platform'] = df['platform'].where(df['platform'] != 'NES', 'Nintendo Entertainment System')  # Solo reemplaza 'NES', el resto se mantiene igual

(df['na_sales'] > 0) | (df['eu_sales'] > 0) #si es mayor sera true, si es menor sera false

df[['na_sales', 'eu_sales']] = df[['na_sales', 'eu_sales']].where((df['na_sales'] > 0) | (df['eu_sales'] > 0), None)

rare_publishers = ['Red Flagship', 'Max Five', '989 Sports']
df['publisher'] = df['publisher'].where(~df['publisher'].isin(rare_publishers), 'other')

df['genre'] = df['genre'].where(~df['genre'].isin(genres), 'Misc')

df['year_released'] = df['year_released'].where(~((df['name'] == 'Tetris') & (df['year_released'].isna())),1984)


