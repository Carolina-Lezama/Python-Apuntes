#------------IMPORTAR PANDAS----------------
import pandas as pd

#------------ABRIR ARCHIVOS CSV----------------
#Se guardan en una variable
df = pd.read_csv('/datasets/music.csv', sep=',') #sep por defecto es , pero puede cambiar como por | ; etc. es la forma en que estan separados los datos
                                        #sep='\\t' #para tabulaciones

column_names = ['user_id', 'genre', 'Artist', 'track', 'total play']
df = pd.read_csv('/datasets/music.csv', header=None , names=column_names) #por si no hay cabeceras, no se tomen filas como cabecera, no tendremos cabeceras por defecto; va junto con names

df = pd.read_csv('/datasets/music.csv', decimal=',') #volver a los numeros 9,1 a 9.1 para que se tomen como flotantes

#------------ABRIR ARCHIVOS EXCEL---------------
df = pd.read_excel('/datasets/online.xlsx') #solo trae la primera hoja por defecto

df = pd.read_excel('/datasets/online.xlsx', sheet_name='hoja_2')# cargar una hoja por nommbre
dfs = pd.read_excel('/datasets/product.xlsx', sheet_name=['product_categories', 'reviews']) #cargar lista de hojas por nombre

df = pd.read_excel('/datasets/online.xlsx', sheet_name=1) #cargar hoja por indice
dfs = pd.read_excel('/datasets/product.xlsx', sheet_name=[0, 2]) #cargar lista de hojas por indice, comienzan en 0

dfs = pd.read_excel('/datasets/product.xlsx', sheet_name=None)# cargar todas las hojas

#------------GUARDAR RUTAS ---------------
file_path = '/datasets/diabetes.csv'
df = pd.read_csv(file_path)

#------------FUNCION LOC----------------
#Existe la notación abreviada y la completa
#Completa:
# indice y columna
df.loc[4, 'genre'] #una celda
df.loc[:, 'genre'] #una columna
df.loc[:, ['genre', 'Artist']] #varias columnas no consecutivas
df.loc[:,'total play': 'genre'] #Múltiples columnas consecutivas (slice)
df.loc[5, 'total play': 'genre'] #una fila y varias columnas consecutivas
df.loc[1] #una fila completa por indice
df.loc[1:] #desde la fila 1 hasta el final slice de filas
df.loc[:3] #desde el inicio hasta la fila 3 slice de filas
df.loc[2:5] #desde la fila 2 hasta la fila 5 slice de filas
df.loc['state 1': 'state 3', 'flower': 'insect'] #slice de indices, slice de columnas
df.loc['state 1': 'state 3', 'flower'] #slice de indices, de una columna
df.loc[[0, 1], ['col_name_1', 'col_name_2']] #no funciona como rango, devuelve las filas 0 y 1, y las columnas especificadas

#Abreviada:
df['genre'] #columna(s)
df[['genre', 'Artist']]

df[1:] #fila(s)
df[:3] 
df[2:5]

#------------CREAR UN DATAFRAME----------------
sales_data = [
    ['Laptop' , 'North America', 120 , 120000],
    ['Smartphone' , 'Europe' , 340 ,170000],
    ['Tablet' , 'Asia' , 210 , 63000],
    ['Headphones' , 'South America' , 150 , 45000],
    ['Smartwatch' , 'Africa' , 95 , 28500]
]
values = ['object', 'capital',"price", "cantite"]
sales = pd.DataFrame(data=sales_data, columns=values)

states  = ['Alabama', 'Alaska', 'Arizona', 'Arkansas']
flowers = ['Camellia', 'Forget-me-not', 'Saguaro cactus blossom', 'Apple blossom']
insects = ['Monarch butterfly', 'Four-spotted skimmer dragonfly', 'Two-tailed swallowtail', 'European honey bee']
index   = ['state 1', 'state 2', 'state 3', 'state 4']

df = pd.DataFrame({'state': states, 'flower': flowers, 'insect': insects}, index=index) #crear columnas con listas y asignar indices personalizados

#------------CONTAR, OBTENER MEDIA Y SUMAR----------------
pop_df =df[df['genre'] == 'Pop'] #aislar datos a cierto tipo, solo tendra las filas que cumplan la condicion

duracion = pop_df[pop_df['user_id'] == '123456']['total play'].sum() #Usas otra condicion para datos aun mas especificos, se toma el campo y se aplica la funcion
promedio = pop_df['total play'].mean() #media general, en el caso de arriba se toma un usuario como especifico 
solo_una = df[df['genre'] == 'Pop']['total play'].mean() #o cualquier operacion
contar = pop_df['total play'] > 30 ['total play'].count() #condicion que se debe cumplir, se cuenta aquellos que cumplen la condicion

#------------FUNCIONES EN PANDAS BASE----------------
print(df.head(2)) #primeras 5 filas por defecto
print(df.tail(6)) #ultimas 5 filas por defecto
print(df.sample(5)) #muestra aleatoria de 5 filas

print(df.dtypes) #columnas y tipos de datos
print(df.shape) #numero de filas
print(df.columns) #nombres de las columnas

todo = df.info() #informacion general del DataFrame

#------------INDEXACION LOGICA O BOOLEANA----------------
#Cumplir condiciones, dan booleanos si se cumple o no, devolvera aquellas filas que sean true

#Notacion mezclada:
df.loc[df['genre'] == 'Rock'] #completa usando abreviada dentro

#Notacion abreviada:
df[df['genre'] == 'Rock'] #abreviada /abreviada
df[df['total play'] > 90] 
desktop_data = df[df['device_type'] == 'desktop']
mobile_data =  df[df['device_type'] == 'mobile']

ventas_filtradas = df[df['sucursal'] == 'Centro'] #condicional doble, separada
ventas_filtradas[ventas_filtradas['precio_unitario'] > 25]

ventas_filtradas = df[(df['sucursal'] == 'Centro') & (df['precio_unitario'] > 25)] #condicional doble, unida

#Notacion completa:
df.loc[df.loc[:,'genre'] == 'Rock'] #completa /completa

#------------ CAMBIAR NOMBRE DE COLUMNAS----------------

new_columns ={ #nombre anterior : nuevo nombre
    '  user_id': 'user_id', 
    'total play': 'total_play',
    'Artist': 'artist'
    }
df = df.rename(columns = new_columns)
df.rename(columns=new_columns, inplace=True) #sin reasignar
df.rename(columns={'  user_id': 'user_id', 'total play': 'total_play', 'Artist': 'artist' }, inplace=True)

columna_nombre = []
for viejo_nombre in df.columns: #ciclo para cambiar nombres sin hacerlo manual
    nuevo_nombre = viejo_nombre.strip().lower().replace(" ","_")
    columna_nombre.append(nuevo_nombre)
df.columns = columna_nombre

#------------ VALORES DUPLICADOS ----------------
df.duplicated().sum() #devuelve filas duplicadas (booleano), se puede sumar para ver cuantos duplicados hay
df = df.drop_duplicates().reset_index(drop=True) #elimina duplicados, resetear los indices, borrar anterior lista de indices
df.drop_duplicates(inplace=True) #elimina duplicados sin reasignar

df[df.duplicated(keep=False)] #devuelve la posiciones de los duplicados incluto la primera aparicion

df.drop_duplicates(subset='col_1') #Busca y elimina duplicados solo considerando la columna
df.drop_duplicates(subset=['col1', 'col2']) #Busca y elimina duplicados basandose en varias columnas

#------------UNIQUE Y NUNIQUE---------------
df['name'].unique() #valores unicos en una columna, sirve para verificar duplicados por mala escritura, etc. devuelve array
df['name'].nunique() #da el numero de valores unicos, devuelve entero

#------------ CONTAR VALORES----------------
df['col_1'].value_counts(dropna=False) #Incluira los valores None o NaN si es False
df['col_1'].value_counts(ascending=True) #contar valores unicos en una columna, dira cuantas veces aparece cada uno










#------------ AGRUPAMIENTO DE DATOS ----------------
#primero es el argumento o agrupamiento por el que se dividira, el segundo son las columnas a las que se les aplicara la funcion 
#dividir los datos - aplicar una funcion a cada grupo - combinar los resultados para cada grupo
grouped_stage_count = df.groupby('Stage')['Digimon'].count()
grouped_stage_sum = df.groupby('Stage')['Lv 50 HP'].sum()
grouped_stage_mean = df.groupby('Stage')['Lv50 Spd'].mean()
mean_score = df.groupby('genre')['critic_score'].mean()
#cambia el índice de fila de los datos a las clave  s por las que estamos agrupando.

groupby_data = df.groupby(['genre', 'platform'])['eu_sales'].mean()
print(groupby_data)

grp = df.groupby(['platform', 'genre']) #agrupar por varios campos 
print(grp['critic_score'].mean())

df_grouped = df.groupby('score_categorized') #este por si solo no es visible, debe tener un [] para que devuelva datos

df_sum = df_grouped['na_sales'].sum()

#------------ ORDENAR DATOS ----------------
df.sort_values(by='Stage',ascending=True) #por valor numerico o string, ordena toda la tabla por cierta columna
df_year_of_release = df['year_of_release'].value_counts().sort_index() #aplicado a indices

df['radius'].sort_values() #solo ordena esa columna

metal_ordenado = df[df['genre'] == 'metal'].sort_values(by='total_play', ascending=False) #filtrado y ordenado segun otro parametro

exo_small_14 = df[df['radius'] < 1] #doble filtrado y finalmente ordenamiento
exo_small_14 = exo_small_14[exo_small_14['discovered'] == 2014]
exo_small_14 = exo_small_14.sort_values(by='radius', ascending=False) #se devuelve un objeto

#------------ RELLENAR VALORES CATEGORICOS AUSENTES----------------
df = pd.read_csv('/datasets/music_log_chpt_11.csv', keep_default_na=False) #vuelve cadenas vacias en vez de NaN

#------------ METODOS A STRINGS DE FILAS----------------
df['col_1'] = df['col_1'].str.lower() #convierte a minusculas los strings de una columna, con reasignacion
df['item_lowercase'] = df['item'].str.lower() #asignacion a nueva columna

df['item_modified'] = df['item'].str.replace('GB', 'gb') #sin reasignacion a la original

df = df.rename(columns={'userid': 'user_id'})

df['city'] = df['city'].str.strip().str.title() #title hace mayuscula la primera letra de cada palabra

df['genre'] = df['genre'].replace('ranchera','rancheras') #remplaza un valor por otro
df['source'] = df['source'].replace('','email')

#------------ FUNCION PARA REMPLAZAR ERRORES EN DATOS ----------------
def replace_wrong_values(df, column, wrong_values, correct_value):
    for wrong_value in wrong_values:
        df[column] = df[column].replace(wrong_values, correct_value)
    return df
duplicated_names = ['Rafael Nadad', 'Rafa Nadal']
name="Rafael Nadal"
replace_wrong_values(df, 'name', duplicated_names, name)

#------------ FILTRAR CON NaN----------------
df[~df['source'].isna()] #filtra los que no son NaN, el ~ es un NOT; muestra los que son valores presentes

#------------DESCRIBE---------------
#resumen estadístico detallado para cada una de las columnas; por defecto solo numericas
print(df.describe()) 
print(df.describe(include='object')) #count igual a valores no nulos, unique valores unicos, top valor mas frecuente, freq frecuencia del valor mas frecuente
print(df.describe(include='all')) #todas las columnas; nos devolverá valores NaN para aquellas estadísticas que no sean aplicables al tipo de datos de la columna

#------------ BUSCAR VALORES AUSENTES ----------------
df.isna().sum() #si es true es ausente, al final se suman los ausentes.

#------------ LLENAR VALORES AUSENTES ----------------
df['genre'].fillna(0, inplace = True) #rellena los ausentes con 0
df['email'] = df['email'].fillna(value='') #value esepcidifca con que valor quieres reemplazar los valores ausentes
df['col'].fillna('no_info', inplace=True)

#------------ ELIMINAR VALORES AUSENTES ----------------
df['my column'] = df['my column'].dropna() #elimina ausentes de una columna especifica

df.dropna(subset=['genre','total play'], inplace=True) #elimina filas con minimo un ausente en la lista de columnas

df = df.dropna(axis = 'columns') #eliminar la columna si tiene al menos un ausente
df = df.dropna(axis = 'rows') #eliminar la fila si tiene al menos un ausente

#------------ ELIMINAR FILAS O COLUMNAS QUE YO INDIQUE----------------
df = df.drop(labels=['notes', 'puede_ser_una_o_varias'], axis='columns') #eliminar una columna especifica
df = df.drop(labels=0, axis='columns') #eliminar una columna por indice
#ambos aceptan inplace = True

#------------OBJETO SERIES Y DETERMINAR EL INDEX---------------
oceans = pd.Series(['Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic'], index=['A', 'B', 'C', 'D', 'E']) #Forma 1, puedo usar int o string
oceans.index = [1, 2, 3, 4, 5] #Forma 2

#------------INDEXACION ILOC[]----------------
#usa enteros, se puede usar indexacion negativa
df.iloc[3, 2] #4ª fila y la 3ª columna
df.iloc[[0, 2], 1:]
df[df['publisher'] == 'other'].iloc[:,1:5]
#------------CAMBIAR INDICES CON SET_INDEX() PARA DATAFRAME----------------
df = df.set_index('state') #toma una columna existente de un DataFrame y reemplaza el índice con los valores de esa columna, esa columna deja de exisitir en otros lados
df.index.name = None #borrar el nombre de la columna, no tendremos titulo en los index

#------------FILTRADO CON EL METODO query()----------------
#requiere un string(representa la consulta)
df.query("publisher == 'Nintendo'")[['name', 'publisher']].head()
df.query("~(platform in @handhelds)")[['name', 'platform']]#@ dentro se utiliza para hacer referencia a una variable externa
df.query("jp_sales > 1")[['name', 'jp_sales']]
df.query("(platform not in @handhelds)")[['name', 'platform']]
df.query("genre not in @s_genres")[oceans]
df.query('year_of_release >= 2006 and year_of_release <= 2008')
df.query('year_of_release in [2006, 2007, 2008]')

our_list = [2, 5, 10]
df = pd.DataFrame(
    {
        'a': [2, 3, 10, 11, 12],
        'b': [5, 4, 3, 2, 1],
        'c': ['X', 'Y', 'Y', 'Y', 'Z'],
    }
)
print(df.query("a in @our_list")) #devolvió todas las filas de df donde los valores de 'a' están presentes en our_list

our_dict = {0: 10, 3: 11, 12: 12}
print(df.query("a in @our_dict.values()")) #funciona igual con diccionarios

our_series = pd.Series([10, 11, 12])
print(df.query("a in @our_series"))#funciona igual con series; .index para comprobar si los valores están presentes entre los valores del índice

our_df = pd.DataFrame(index=[10, 11, 12]) #usando otro datadrame como filtro, usando indices
print(df.query("a in @our_df.index"))

our_df = pd.DataFrame({'a1': [10, 11, 12]})
print(df.query("a in @our_df.a1"))#usando el nombre de una columna, se usa . en lugar de['']

cols = ['name', 'publisher', 'developer']
df_filtered = df.query("publisher == developer")[cols]

df.query("platform == 'Wii' and genre != 'Sports'").head()

q_string = "na_sales >= 1 or eu_sales >= 1 or jp_sales >= 1"
print(df.query(q_string).head())

cols = ['name', 'developer', 'na_sales', 'eu_sales', 'jp_sales']
q_string = "na_sales > 0 and eu_sales > 0 and jp_sales > 0"
df_filtered = df.query(q_string)[cols]

developers = ['SquareSoft', 'Enix Corporation', 'Square Enix']
cols = ['name', 'developer', 'na_sales', 'eu_sales', 'jp_sales']
q_string = "jp_sales > (na_sales + eu_sales) and jp_sales > 0 and na_sales > 0 and eu_sales > 0 and developer in @developers"
df_filtered = df.query(q_string)[cols]

#------------FILTRADO CON EL METODO isin()----------------
#comprueba si los valores de una columna coinciden con alguno de los valores de otra matriz, conveniente cuando tenemos muchas condiciones que comprobar
#se puede usar ~    
handhelds = ['3DS', 'DS', 'GB', 'GBA', 'PSP'] 
df[df['platform'].isin(handhelds)][['name', 'platform']] #comprueba si los valores de la columna 'platform' son iguales a algún valor de la lista handhelds
df_filtered = df[~df['genre'].isin(s_genres)][cols]
ventas_filtradas = ventas[ventas['producto'].isin(productos_más_vendidos)]
((df['critic_score'] >= 90) | (df['user_score'] >= 9)) & (df['genre'].isin(['Role-Playing', 'Strategy', 'Puzzle']))
df[df['year_of_release'].isin([2006, 2007, 2008])]

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