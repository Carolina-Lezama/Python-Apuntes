#-----------¿Qué es la correlacion?--------------
#mide qué tan fuerte es la relación lineal entre dos variables numéricas; "¿Cuando una variable cambia, la otra también cambia de manera predecible?"
#0.7 a 1.0: Correlación fuerte
# 0.3 a 0.7: Correlación moderada
# 0.0 a 0.3: Correlación débil
#+1: Correlación positiva perfecta: - Cuando una variable aumenta, la otra también aumenta - Ejemplo: Más ejercicio → Mejor salud
#0: Sin correlación: - No hay relación predecible entre las variables - Ejemplo: Color de zapatos y calificaciones
#-1: Correlación negativa perfecta: - Cuando una variable aumenta, la otra disminuye - Ejemplo: Más horas viendo TV → Menos tiempo 

#-----------Bucles sobre claves y valores en diccionarios--------------
# útiles cuando quieres realizar operaciones solo en las claves o los valores, o cuando quieres comprobar si una clave o un valor en particular está presente 
financial_info = {
    'American Express': 93.23,
    'Boeing': 178.44,
    'Coca-Cola': 45.15,
    'Walt Disney': 119.34,
    'Nike': 97.99,
    'JPMorgan':96.27,
    'Walmart': 130.68 
}

for key in financial_info.keys(): #llaves ''
    print(key)
for value in financial_info.values(): #valores .
    print(value)
for key, value in financial_info.items(): #extrae ambos
    print(key, value)

#-----------Diccionarios de listas--------------
bus_schedule = {
    '72': ['8:00', '12:00', '17:30'],
    '26': ['9:30', '15:00'],
    '17': ['7:30', '12:30', '15:30']
}
# iteramos sobre las claves y los valores del diccionario
for route, times in bus_schedule.items():
    # iteramos sobre los valores de la lista
    for time in times:
        # mostramos la ruta y su horario correspondiente
	      print(f"Ruta {route} - Hora {time}")

#-----------Lista de diccionarios--------------
movies_table = [
    {'movie_name':'The Shawshank Redemption', 'country':'USA', 'genre':'drama', 'year':1994, 'duration':142, 'rating':9.111},
    {'movie_name':'The Godfather', 'country':'USA', 'genre':'drama, crime', 'year':1972, 'duration':175, 'rating':8.730},
    {'movie_name':'The Dark Knight', 'country':'USA', 'genre':'fantasy, action, thriller', 'year':2008, 'duration':152, 'rating':8.499}
]
# ahora accedemos a la columna por su nombre:
print(movies_table[2]['movie_name'])

order = [
    {
        'item': 'Margherita pizza',
        'category': 'pizza',
        'quantity': 2,
        'price': 9
    },
    {
        'item': 'Ham pizza',
        'category': 'pizza',
        'quantity': 1,
        'price': 12
    },
    {
        'item': 'Pepsi 1 l',
        'category': 'beverages',
        'quantity': 3,
        'price': 2
    }
]
# variable del precio total del pedido
total_price = 0 
# iterar sobre cada diccionario de la lista
for item in order: 
# añadir a la variable el precio del elemento multiplicado por la cantidad
    total_price += item['price'] * item['quantity']
print(total_price)

# variable para almacenar el resultado
filtered_order = [] 
for item in order: # iterar sobre cada diccionario de la lista
	if item['category'] == 'pizza': # si la categoría es pizza...
		filtered_order.append(item) # agregamos el diccionario a la lista filtered_order
# mostramos la lista filtrada de diccionarios
print(filtered_order) 

#-----------Funciones: parámetros y valores predeterminados--------------
# añade un valor por defecto para eggs_number al parámetro
def omelet(cheese, eggs_number=2): #primero obligatorio, luego opcionales
    result = '¡El omelet está listo! Huevos utilizados: ' + str(eggs_number)
    if cheese == True:
        result = result + ', con queso'
    else:
        result = result + ', sin queso'
    return result
print(omelet(False))

def filter_by_genre(data, genre='drama'):
    filtered_result = []
    for row in data:
        if genre in row[3]:
            filtered_result.append(row)
    return filtered_result

#-----------Funciones: valores de retorno--------------
def find_area_and_perim(side1, side2):
    area = side1 * side2
    perimeter = 2 * (side1 + side2)
    return area, perimeter
# descomprime el resultado de la función
rec_area, rec_perimeter = (find_area_and_perim(7, 3))
print(f'El área del rectángulo es {rec_area}, el perímetro es {rec_perimeter}')

#-----------Pandas: el objeto Series--------------
#Cuando recuperas varias columnas o filas de una tabla, obtienes una nueva tabla que es un DataFrame
import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df[['genre', 'Artist']]
print(type(part_df)) #tipo DataFrame

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv') #solo una columna
part_df = df['Artist']
print(type(part_df)) #tipo Series

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df['Artist']
print(part_df.name)#el atributo name adopta el nombre de la columna

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df['Artist']
print(part_df.size) #tamaño del series

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
# obtenemos un objeto Series a partir del DataFrame
artist = df['Artist']
print(artist[0])# obtenemos una celda de un Series con una sola coordenada

#	Notación completa
total_play.loc[7]	
total_play.loc[[5, 7, 10]]	
total_play.loc[5:10] 	
total_play.loc[1:]	
total_play.loc[:3] 

#	Notación abreviada
total_play[7]
total_play[[5, 7, 10]]
total_play[5:10]
total_play[1:]
total_play[:3]	

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
total_play = df['total play']
lower_20 = total_play < 20 #El resultado muestra el índice de cada canción del Series junto con el valor booleano 
df_lower20_only = df[lower_20]
print(df_lower20_only)

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
total_play = df['total play']
lower_20 = total_play < 20
series_lower20_only = total_play[lower_20] #filtrar el dataset
print(series_lower20_only)

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
tracks = df['track']
track_check = tracks == 'Andrew Paul Woodworth'

import pandas as pd
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
genre = df['genre']
pop_genre_check = genre == 'pop'
pop_df = df[pop_genre_check]
print(pop_df)

#-----------Pandas: estadística descriptiva--------------
import pandas as pd
df = pd.read_csv('/datasets/music_log_processed.csv')
print(df['total_play'].max()) #maximo
print(df[df['total_play'] == df['total_play'].max()]) #indexacion logica del maximo

import pandas as pd
df = pd.read_csv('/datasets/music_log_processed.csv')
pop_tracks = df[df['genre'] == 'pop'] #extraer todas las filas correspondientes a canciones pop
pop_tracks = pop_tracks[pop_tracks['total_play'] > 30] #eliminamos todas las filas donde la columna 'total_play' tiene menos de 30 segundos
max_dur = pop_tracks['total_play'].max() #a reproducción más larga de una pista de pop  
print(max_dur)

df_drop_skip = df[df['total_play'] > 30]
print(df_drop_skip['total_play'].min())#minimo

print(df_drop_skip[df_drop_skip['total_play'] == df_drop_skip['total_play'].min()])

print(df['total_play'].median()) #mediana, valor en medio de la lista ordenada

df_drop_skip = df[df['total_play'] > 30]
print(df_drop_skip['total_play'].median())

print(df_drop_skip['total_play'].mean()) #media, valor promedio de un conjunto de datos.

#Si la media fuera mucho mayor o menor, eso podría indicar la presencia de valores atípicos. 
#si ves una diferencia significativa entre la media y la mediana, esta indica que hay muchos valores atípicos

df = pd.read_csv('/datasets/music_log_processed.csv')
pop_tracks = df[df['genre'] == 'pop']
pop_tracks = pop_tracks[pop_tracks['total_play'] > 30]
pop_mean = pop_tracks['total_play'].mean()
pop_median = pop_tracks['total_play'].median()
print(pop_mean)
print(pop_median)