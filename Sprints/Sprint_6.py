#-----------¿Qué es la correlacion?--------------
# Mide qué tan fuerte es la relación lineal entre dos variables numéricas; "¿Cuando una variable cambia, la otra también cambia de manera predecible?"

# 0.7 a 1.0: Correlación fuerte
# 0.3 a 0.7: Correlación moderada
# 0.0 a 0.3: Correlación débil

# +1: Correlación positiva perfecta: - Cuando una variable aumenta, la otra también aumenta 
    # Ejemplo: Más ejercicio → Mejor salud
#  0: Sin correlación: - No hay relación predecible entre las variables 
    # Ejemplo: Color de zapatos y calificaciones
# -1: Correlación negativa perfecta: - Cuando una variable aumenta, la otra disminuye 
    # Ejemplo: Más horas viendo TV → Menos tiempo 

#-----------Bucles sobre claves y valores en diccionarios--------------
# Utiles para realizar operaciones solo en las claves o los valores o comprobar si una clave o un valor en particular está presente 

financial_info = {
    'American Express': 93.23,
    'Boeing': 178.44,
    'Coca-Cola': 45.15,
    'Walt Disney': 119.34,
    'Nike': 97.99,
    'JPMorgan':96.27,
    'Walmart': 130.68 
}

for key in financial_info.keys(): # Llaves(izquierda)
    print(key)

for value in financial_info.values(): # Valores(derecha)
    print(value)

for key, value in financial_info.items(): # Ambos
    print(key, value)

#-----------Diccionarios de listas--------------
bus_schedule = {
    '72': ['8:00', '12:00', '17:30'],
    '26': ['9:30', '15:00'],
    '17': ['7:30', '12:30', '15:30']
}

# iteramos sobre claves y valores 
for route, times in bus_schedule.items():
    # iteramos sobre los valores de la lista
    for time in times:
	    print(f"Ruta {route} - Hora {time}")

#-----------Lista de diccionarios--------------
movies_table = [
    {'movie_name':'The Shawshank Redemption', 'country':'USA', 'genre':'drama', 'year':1994, 'duration':142, 'rating':9.111},
    {'movie_name':'The Godfather', 'country':'USA', 'genre':'drama, crime', 'year':1972, 'duration':175, 'rating':8.730},
    {'movie_name':'The Dark Knight', 'country':'USA', 'genre':'fantasy, action, thriller', 'year':2008, 'duration':152, 'rating':8.499}
]

# Accedemos a un diccionario de la lista, luego accedemos a una columnas del diccionario
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
        'item': 'Pepsi 1l',
        'category': 'beverages',
        'quantity': 3,
        'price': 2
    }
]

# Declaracion de variable: precio total del pedido
total_price = 0 

# Iterar sobre cada diccionario de la lista
for item in order: 
    total_price += item['price'] * item['quantity']
print(total_price)


filtered_order = [] 

# Iterar sobre cada diccionario de la lista
for item in order: 
	if item['category'] == 'pizza': # si la categoría es pizza...
		filtered_order.append(item) # agregamos el diccionario a la lista 
print(filtered_order) 

#-----------Funciones: parámetros y valores predeterminados--------------

# Valor por defecto para el parámetro eggs_number 
def omelet(cheese, eggs_number=2): # Obligatorio, Opcional
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
# Desempaqueta el resultado de la función
rec_area, rec_perimeter = (find_area_and_perim(7, 3))
print(f'El área del rectángulo es {rec_area}, el perímetro es {rec_perimeter}')

#-----------Impotar libreria pandas--------------
import pandas as pd

#-----------Pandas: el objeto Series--------------

df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df[['genre', 'Artist']]
print(type(part_df)) # Tipo DataFrame

df = pd.read_csv('/datasets/music_log_chpt_11.csv') 
part_df = df['Artist']
print(type(part_df)) # Tipo Series(solo una columna)

df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df['Artist']
print(part_df.name)

df = pd.read_csv('/datasets/music_log_chpt_11.csv')
part_df = df['Artist']
print(part_df.size) 

df = pd.read_csv('/datasets/music_log_chpt_11.csv')
artist = df['Artist']
print(artist[0])

#	Notación completa
part_df.loc[7]	
part_df.loc[[5, 7, 10]]	
part_df.loc[5:10] 	
part_df.loc[1:]	
part_df.loc[:3] 

#	Notación abreviada
part_df[7]
part_df[[5, 7, 10]]
part_df[5:10]
part_df[1:]
part_df[:3]	

#-----------Filtrado de celdas--------------
df = pd.read_csv('/datasets/music_log_chpt_11.csv')
total_play = df['total play']
lower_20 = total_play < 20 # Mascara booleana
df_lower20_only = df[lower_20]
print(df_lower20_only)


df = pd.read_csv('/datasets/music_log_chpt_11.csv')
tracks = df['track']
track_check = tracks == 'Andrew Paul Woodworth'
resultado = df[track_check]
print(resultado)


df = pd.read_csv('/datasets/music_log_chpt_11.csv')
genre = df['genre']
pop_genre_check = genre == 'pop'
pop_df = df[pop_genre_check]
print(pop_df)

#-----------Pandas: estadística descriptiva--------------
# Maximo
df = pd.read_csv('/datasets/music_log_processed.csv')
print(df['total_play'].max()) 
print(df[df['total_play'] == df['total_play'].max()]) # Indexacion logica del maximo

# Filtrado por condicionales 
df = pd.read_csv('/datasets/music_log_processed.csv')
pop_tracks = df[df['genre'] == 'pop'] 
pop_tracks = pop_tracks[pop_tracks['total_play'] > 30] 
max_dur = pop_tracks['total_play'].max() # Pista mas larga
print(max_dur)

# Minimo
df_drop_skip = df[df['total_play'] > 30]
print(df_drop_skip['total_play'].min())

print(df_drop_skip[df_drop_skip['total_play'] == df_drop_skip['total_play'].min()])

# Mediana (valor en medio de la lista ordenada)
print(df['total_play'].median())

df_drop_skip = df[df['total_play'] > 30]
print(df_drop_skip['total_play'].median())

# Media (valor promedio de un conjunto de datos)
print(df_drop_skip['total_play'].mean()) 

# Si la media fuera mucho mayor o menor, eso podría indicar la presencia de valores atípicos. 
# si ves una diferencia significativa entre la media y la mediana, esta indica que hay muchos valores atípicos

# Comparacion mediana y media
df = pd.read_csv('/datasets/music_log_processed.csv')
pop_tracks = df[df['genre'] == 'pop']
pop_tracks = pop_tracks[pop_tracks['total_play'] > 30]
pop_mean = pop_tracks['total_play'].mean()
pop_median = pop_tracks['total_play'].median()
print(pop_mean)
print(pop_median)