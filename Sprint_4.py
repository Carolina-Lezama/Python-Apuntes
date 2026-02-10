#------------CONVERTIR A OTRO TIPO DE DATOS---------------
df['column'] = df['column'].astype('int') #no sirve de float a int

df['col1'] = pd.to_numeric(df['col1'], errors='coerce')  #string a entero
    #errors='raise' (por defecto): se detiene y lanza un error 
    #errors='coerce': reemplaza los valores no numéricos con NaN
    #errors='ignore': deja los valores originales

#------------VERIFICAR DATOS---------------
#true si es todos son iguales, false si uno es distinto
np.array_equal(df['col1'], df['col1'].astype('int')) #saber si todos son enteros, aunque el tipo de datos sea otro

df['SSN'] = df['SSN'].str.replace('-', '').astype(int) # Garantía de que la columna 'SSN' no tiene guiones y su conversión a entero
df['platform'] = df['platform'].astype('category')

#------------IMPORTAR NUMPY---------------
import numpy as np

#------------CAMBIAR DATO A DATETIME---------------
df['col1'] = pd.to_datetime(df['col1'], format='%Y-%m-%dT%H:%M:%SZ') #%y año en 2 digitos, %I formato de 12h, 

string_date = '5/13/13 12:04:00'
fecha = pd.to_datetime(string_date, format='%m/%d/%y %H:%M:%S') #formato estadounidense

raw_date = '20-12-2002Z04:31:00'
clean_date = pd.to_datetime(raw_date, format='%d-%m-%YZ%H:%M:%S')

position['timestamp'] = pd.to_datetime(position['timestamp'], format='%Y-%m-%dT%H:%M:%S')

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%dT%H:%M:%SZ')

dt_months = position['timestamp'].dt.month
print(dt_months.head())

#------------FUNCIONES A FECHAS---------------
df['col1'].dt.round('H') #redondear las horas

#------------ACCEDER A LOS ATRIBUTOS DE UNA FECHA---------------
df['Day'] = df['InvoiceDate'].dt.day
print(df[['InvoiceDate', 'Day']].head(5))
    # .dt.year → año
    # .dt.month → mes
    # .dt.day → día
    # .dt.hour → hora
    # .dt.minute → minuto
    # .dt.weekday → día de la semana (0 = lunes, 6 = domingo)
    # .dt.date → solo la fecha, sin la hora

#------------ZONAS HORARIAS--------------
df['InvoiceDate'] = df['InvoiceDate'].dt.tz_localize('UTC') #asigna una zona horaria a una columna de tipo datetime.
df['InvoiceDate_NYC'] = df['InvoiceDate'].dt.tz_convert('America/New_York') #convierte una columna con zona horaria a otra zona horaria diferente; el valor se remplaza con la lista permitida
#https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List

dt_toronto = position['timestamp'].dt.tz_localize('America/Toronto')
print(dt_toronto.head())

df['datetime'] = df['datetime'].dt.tz_localize('CET').dt.tz_convert('EST')

#------------CREAR COLUMNAS EN BASE A VALORES DE OTRAS---------------
df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales']
print(df['total_sales'].head())

df['eu_sales_share'] = df['eu_sales'] / df['total_sales']

df['is_nintendo'] = df['publisher'] == 'Nintendo'
print(df['platform'].str.lower().isin(['gb', 'wii']).head())

df['average_sales'] = df[['na_sales', 'eu_sales', 'jp_sales']].mean(axis=1) #axis=0 (por defecto): calcula el promedio columna por columna.axis=1: calcula el promedio fila por fila

df['total_sales'] = df['quantity'] * df['price']

#------------CREAR COLUMNAS CON APPLY()---------------
# toma valores de una columna DataFrame y les aplica una función.
df['territory_group'] = df.apply(post_code_and_address_to_territory_group, axis=1)

def era_group(year):
    if year < 2000:
        return 'retro'
    elif year < 2010:
        return 'modern'
    elif year >= 2010:
        return 'recent'
    else:
        return 'unknown'

df['era_group'] = df['year_of_release'].apply(era_group)
print(df['era_group'].value_counts())

def score_group(score):
    if score < 60:
        return 'low'
    elif score < 80:
        return 'medium'
    elif score >= 80:
        return 'high'
    else:
        return 'no score'
df['score_categorized'] = df['critic_score'].apply(score_group)

def avg_score_group(row):
    critic_score = row['critic_score']
    user_score = row['user_score']
    avg_score = (critic_score + user_score * 10) / 2

    if avg_score < 60:
        return 'low'
    elif 60 <= avg_score <= 79:
        return 'medium'
    elif avg_score >= 80:
        return 'high'
col_names = ['critic_score', 'user_score']
test_low  = pd.Series([10, 1.0], index=col_names)
test_med  = pd.Series([65, 6.5], index=col_names)
test_high = pd.Series([99, 9.9], index=col_names)
rows = [test_low, test_med, test_high]
for row in rows:
    print(avg_score_group(row))

#------------METODO AGG()---------------
#usa un diccionario como entrada, claves son nombres de columnas y valores  son las funciones
agg_dict = {'critic_score': 'mean', 'jp_sales': 'sum'}
grp = df.groupby(['platform', 'genre'])
print(grp.agg(agg_dict))

def double_it(sales):
    sales = sales.sum() * 2 # multiplica la suma anterior por 2
    return sales
agg_dict = {'jp_sales': double_it}
grp = df.groupby(['platform', 'genre'])
print(grp.agg(agg_dict))


df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales']
grp = df.groupby('genre')
agg_dict = { 'total_sales': 'sum', 'na_sales': 'mean', 'eu_sales': 'mean', 'jp_sales': 'mean'}
genre = grp.agg(agg_dict)

#------------TABLAS DINAMICAS---------------
pivot_data = df.pivot_table(index='genre', #la columna cuyos valores se vuelven los indices
                            columns='platform', # la columna cuyos valores se vuelven las columnas
                            values='eu_sales', #los datos que van dentro de la tabla
                            aggfunc='sum' #funcion de agregacion
)

#------------COMBINAR DATAFRAMES CON CONCAT()---------------
mean_score = df.groupby('publisher')['critic_score'].mean() #ambos comparten mismo indice
df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales']
num_sales = df.groupby('publisher')['total_sales'].sum()
df_concat = pd.concat([mean_score, num_sales], axis='columns') # axis para asegurarnos de que se combinaran como columnas.
df_concat.columns = ['avg_critic_score', 'total_sales'] #renombrar columnas
# index=0 concatenará filas y index=1 concatenará columnas
#Para concatenar filas  axis='index'

rpgs = df[df['genre'] == 'Role-Playing']
platformers = df[df['genre'] == 'Platform']
df_concat = pd.concat([rpgs, platformers])
print(df_concat[['name', 'genre']])

total_sales = df.groupby('platform')['total_sales'].sum()
num_pubs = df.groupby('platform')['publisher'].nunique()
platforms = pd.concat([total_sales, num_pubs], axis='columns')
platforms.columns = ['total_sales', 'num_publishers']
print(platforms)

#------------COMBINAR DATAFRAMES CON MERGE()---------------
#afecta a la cantidad de datos, combinar entradas que tienen los datos
first_pupil_df = pd.DataFrame(
    {
        'author': ['Alcott', 'Fitzgerald', 'Steinbeck', 'Twain', 'Hemingway'],
        'title': ['Little Women',
                  'The Great Gatsby',
                  'Of Mice and Men',
                  'The Adventures of Tom Sawyer',
                  'The Old Man and the Sea'
                 ],
    }
)
second_pupil_df = pd.DataFrame(
    {
        'author': ['Steinbeck', 'Twain', 'Hemingway', 'Salinger', 'Hawthorne'],
        'title': ['East of Eden',
                  'The Adventures of Huckleberry Finn',
                  'For Whom the Bell Tolls',
                  'The Catcher in the Rye',
                  'The Scarlett Letter'
                 ],
    }
)

both_pupils = first_pupil_df.merge(second_pupil_df, on='author')
print(both_pupils)#aunque incluye todas las columnas  pero solo se conservan las filas con datos compartidos
#si para una columna hay mas de un dato se hara un dato_x dato_y para poder asignar ambos al mismo nombre de columna

both_pupils = first_pupil_df.merge(second_pupil_df, on='author', how='outer') #conservan todas las filas, donde no hay coincidencia  se pone valor ausente

both_pupils = first_pupil_df.merge(second_pupil_df, on='author', how='left',suffixes=('_1st_student', '_2nd_student'))
# todos los valores del DataFrame izquierdo están presentes en el DataFrame fusionado. Los valores del DataFrame derecho  solo se conservan para los valores que coinciden con la columna especificada en el izquierdo

both_pupils = first_pupil_df.merge(second_pupil_df,
                                   left_on='authors',
                                   right_on='author'
                                  )
print(both_pupils.drop('author', axis='columns')) #eliminar la información duplicada    

df_merged = df_members.merge(
    df_orders,
    left_on='id', #si los combres de columnas son distintas en cada frame
    right_on='user_id',
    how='inner',
    suffixes=('_member', '_order')
)
df_merged = df_merged.drop('user_id', axis='columns')
print(df_merged)

#------------Crear gráficas en Python con matplotlib---------------
#------------Importacion de librerias ---------------
import pandas as pd
from matplotlib import pyplot as plt

#------------Método plot() y plt.show()---------------
#plot():crear una gráfica en pandas
#show():mostrar  una gráfica.
df = pd.DataFrame({'a':[2, 3, 4, 5], 'b':[4, 9, 16, 25]})
df.plot()
plt.show() #Los índices están en el eje X y los valores de las columnas están en el eje Y.

df = pd.DataFrame({'a':[2, 3, 4, 5], 'b':[4, 9, 16, 25]}) #solo se grafica una columna
df['b'].plot()
plt.show()

plt.savefig('myplot.png') #guardar la grafica como imagen png, debes tener un archivo PNG como lo llamaste en cualquier parte donde este tu script

df.plot(x='b', #elegir que columnas van en cada eje, los valores de las columnas se ponen en cada eje
        y='a',
        title='A vs B', #dar titulo
        style='o', #Cambiar el estilo de marcador del gráfico, se marcara con puntos, style='х' seran taches en lugar de puntos
        color='hotpink', #cambiar el color de la linea
        xlabel="Hola, soy B",#cambiar las etiquetas de los ejes
        ylabel="Hola, soy A",
        xlim=[0, 30], # Si pasas una lista de dos números, el primer número será el mínimo y el segundo será el máximo.
        ylim=0, #Si pasas un  número, será el valor mínimo. 
        grid=True, #agregar una cuadrícula al gráfico
        figsize=[10, 4], #tamaño en pulgadas de la grafica
        legend=True #mostrar la leyenda
        ) 

#este grafico no es de linea sino q se marcan las coordenadas
df.plot(style='o-') #mostrar tanto líneas como puntos
#https://matplotlib.org/stable/api/markers_api.html

plt.xlabel("Hola, soy B") # configurando la leyenda x, otra forma
plt.ylabel("Hola, so A") # configurando la leyenda y

#------------Graficos de dispercion---------------
df = pd.read_csv('/datasets/height_weight.csv')

df.sort_values('height').plot(
    x='height',
    y='weight', 
    kind="scatter", #qué tipo de gráfico crear. 'En este caso, 'scatter' 'para un gráfico de dispersión.
    alpha=0.5 #transparencia de los puntos
    )

df.plot(kind="scatter", 
        x='age',
        y='height',
        title='Adult heights',
        alpha=0.36,
        figsize=[8, 6],
        xlabel="Age / years",
        ylabel="Height / inches")
plt.show()

#------------Calcular el coeficiente de correlación---------------
#aplícalo a la columna con la primera variable, y pasa la columna con la segunda variable como un parámetro. No importa el orden de las variables.
print(df['height'].corr(df['weight'])) 
print(df.corr()) #tabla con todas las correlaciones entre todas las variables

#------------Matrices de dispercion--------------
#gráficos de dispersión para cada posible par de parámetros:
#las celdas diagonales son histogramas que muestran la distribución de valores para cada variable individual
pd.plotting.scatter_matrix(df, figsize=(9, 9))
plt.show()

#------------Matrices de correlacion--------------
print(df.corr())

corr_mat = df.corr()
# Extraer los coeficientes de 'male' con las otras columnas
male_corr = corr_mat.loc['male', ['height', 'weight', 'age']]
print(male_corr)


#------------Gráficos de barras-------------
#nos permitirán comparar propiedades numéricas entre categorías
df.plot(x='year',
        kind='bar',
        title='West coast USA population growth',
        xlabel='Year',
        ylabel='Population (millions)')
plt.legend(['CA', 'OR', 'WA'])
#Si no especificas el parámetro y=, se creará automáticamente una barra para cada columna en el DF que no esté en X
plt.show()

cols = ['or_pop', 'wa_pop']
df.plot(x='year',
        y = cols,
        kind='bar',
        title='Pacific Northwest population growth',
        xlabel='Year',
        ylabel='Population (millions)')
plt.legend(['OR', 'WA'])

#Barras verticales (.bar()):
plt.bar(x, y)

#Barras horizontales (.barh()):
plt.barh(y, x)  # Nota que se invierten los parámetros

#------------Histogramas-------------
#el eje X representa la variable y su rango de valores. El eje Y representa la frecuencia de ocurrencia para cada valor.
df = pd.read_csv('/datasets/height_weight.csv')
df.hist(#primera forma, por defecto hace un histograma para cada columna
    column='height', #pedir especificamente una
    bins=30 #numero de contenedores
    ) 
df['height'].hist() #segunda forma

df = pd.read_csv('/datasets/height_weight.csv')
df.plot(kind='hist') #tercera forma
df['height'].plot(kind='hist', bins=30)

df[df['male'] == 1]['height'].plot(kind='hist', bins=30) #juntando dos histogramas en una misma grafica
df[df['male'] == 0]['height'].plot(kind='hist', bins=30, alpha=0.8)
plt.legend(['Male', 'Female']) # leyenda, que sigue el mismo orden trazado anteriormente
plt.show()

df = df.sort_values('male').reset_index(drop=True)
df = df[df.index < 5000]
df.hist(column='height', bins=50) 
plt.show()

df_20s = df[df['age'] < 30]
df_30s = df[(df['age'] >= 30) & (df['age'] < 40)]
df_40s = df[df['age'] >= 40]
df_20s['weight'].plot(kind='hist', bins=20, title='Weight / lbs',ylabel="Frequency",)
df_30s['weight'].plot(kind='hist', bins=20, alpha=0.6)
df_40s['weight'].plot(kind='hist', bins=20, alpha=0.3)
plt.legend(['20s', '30s', '40s'])
plt.show()















df.plot(x='date',
        y='volume',
        title='Historic SBUX volume',
        xlabel="Date",
        ylabel="Volume",
        ylim=[1e6, 7e7],
        legend=False,
        rot=50)
plt.show()
plt.show()


cols = ['open', 'close']
df.plot(title='Historic SBUX price',
        x='date',
        xlabel="Date",
        ylabel="Share price / USD",
        rot=50,
        y=cols)





df.plot(x="t", y="t-1", kind="scatter", alpha=0.1, xlabel="Current value",
        ylabel="Previous value", title="Current vs Previous", figsize=(10,10), rot=45)


