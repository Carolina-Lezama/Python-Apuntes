#------------¿Qué es la minería web o análisis sintáctico?----------------
Esto significa que un análisis relevante debe basarse en datos históricos del tipo de cambio de una fuente externa.
Los analistas pueden enriquecer sus datos complementándolos con datos de Internet. 
Primero rastrean los recursos que pueden ser relevantes, luego recuperan todos los datos necesarios

#------------Otras definiciones----------------
HTML (Hypertext Markup Language)
HTTP (Hypertext Transfer Protocol)
API ("Application Programming Interface")
#------------Protocolos de transferencia---------------
se basa en el principio de "solicitud-respuesta"
un navegador genera una solicitud, luego el servidor la analiza y envía una respuesta
Las reglas para formular solicitudes y respuestas están determinadas por lo que se conoce como protocolo de transferencia
Cuando accedes a un sitio web, tu navegador envía una solicitud HTTP al servidor. El servidor, a su vez, formula una respuesta: el código HTML
GET y POST. El primero solicita datos del servidor, mientras que el segundo los envía
Cuerpo de la solicitud: por ejemplo, el cuerpo de una solicitud POST son los datos que se envían. No todas las solicitudes tienen cuerpo.
Ruta: el segmento de la dirección que sigue al nombre del sitio example.com/hello la ruta es /hello).

#------------Abrir el inspeccionador---------------
Control+Shift+i.

#------------Solicitudes get()---------------
Para obtener datos del servidor
Le pasaremos el enlace como argumento. El método enviará al servidor una solicitud GET, luego procesará la respuesta que reciba y devolverá una respuesta, un objeto

import requests
URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)
print(req.text) #devolvera el texto
print(req.status_code) #respuesta del servidor

#------------expresión regular (regex) ---------------
regla para buscar subcadenas
import re

# Expresión 	    Descripción	                        Ejemplo	            Explicación
# []	            Caracter entre paréntesis	        [a-]	            a & -
# [^…]	            Negación	                        [^a]	            cualquier  excepto "a"
# -	                Rango	[0-9]	                    entre 0 al 9
# .	                Cualquier excepto una nueva línea	a.	                as, a1, a_
# \d (see [0-9])	Cualquier dígito	                a\d	a[0-9]	        a1, a2, a3
# \w	            Cualquier letra, dígito o _	        a\w	                a_, a1, ab
# [A-z] 	        Cualquier letra	                    a[A-z]	            ab
# [A-z ]            incluidos espacios                   a[A-z ]            a , ab, a_
# "[A-z ]+"         cadena entre comillas             "a[A-z ]+"         "a ", "ab", "a_" 
# ?	                0 o 1 instancia	                    a?	                a o nada
# +	                1 o más instancias	                a+	                a o aa, o aaa
# *	                0 o más instancias	                a*	                nada o a, o aa
# ^	                Comienzo de cadena	                ^a	                a1234, abcd
# $	                Fin de cadena	                    a$	                1a, ba

# \+ para buscar el símbolo +
# \. para buscar el símbolo .
# \\ para buscar el símbolo \

# [0-9]: este patrón coincide con cualquier dígito del 0 al 9 (¡solo uno a la vez!). Aplicado a la cadena 155 plus 33, coincidirá con 1, 5, 5, 3 y 3.
# [0-9]+: este patrón coincide con secuencias continuas de dígitos, es decir, números. Aplicado a la cadena 155 plus 33, coincidirá con 155 y 33.
# ^[0-9]+: Aplicado a la cadena 155 plus 33, este patrón producirá una sola coincidencia: 155. solo numeros

# busca un patrón en una cadena, solo devuelve la primera subcadena
string = '"General Slocum" 15 June 1904 East River human factor'
print(re.search('\w+', string))
#Resultado: <re.Match object; span=(1, 8), match='General'>
print(re.search('\w+', string).group()) #solo la subcadena 
print(re.search('"[A-z ]+"', string).group()) #subcadena entre comillas

print(re.split('\d+', string)) #divide la cadena donde aparece el patrón
#Resultado: ['"General Slocum" ', ' June ', ' East River human factor']
print(re.split('\d+', string, maxsplit = 1)) #limita el número de veces que se divide la cadena
#Resultado: ['"General Slocum" ', ' June 1904 East River human factor']

print(re.sub('\d+', '', string))  # buscamos secuencias de 1 o más dígitos y las remplazamos

tion = "Arrived at the station in total frustration"
print(re.findall('[A-z]+tion', tion)) #lista de todas las subcadenas que coinciden con el patrón

string = 'sixty-seven drops of rain'
print(re.findall('\w+-\w+', string))

string = 'sixty-seven drops of rain'
print(len(re.findall('\w+', string))) #5
print(len(re.findall('[\w-]+', string))) #4

#------------Ejecicios de expresiones regulares ---------------
URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text
print(re.search('<title>[A-ü ]+</title>', req_text).group()) #buscar un inicio y cierre, incluyendo lo que este a la mitad, solo la primera ocurrencia

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text
print(re.findall('[A-ü0-9 ]*Mantequilla[A-ü0-9 ]*', req_text)) #buscar todas aquellas ocurrencias que contengan la palabra Mantequilla sea que tengan algo antes o despues

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text
found_products = re.findall('Horizon[ A-ü0-9-%]*', req_text)
print(len(found_products))
print(found_products)

#------------BeautifulSoup---------------
from bs4 import BeautifulSoup
soup=BeautifulSoup(req.text, 'lxml')

# Recorre el árbol y encuentra el primer elemento cuyo nombre se pasa como argumento y lo devuelve junto con las etiquetas y el contenido del elemento.
heading_2=soup.find("h2")
print(heading_2)
print(heading_2.text) #mostrar el contenido sin etiquetas

# Encuentra todas las instancias de un elemento dado en un árbol y devuelve una lista
paragraph = soup.find_all('p')
print(paragraph)

for paragraph in soup.find_all('p'):
    print(paragraph.text)

table = soup.find("table",attrs={"id": "ten_years_first"}) #buscar por atributos, funciona para las 2

heading_table = []  # Lista donde se almacenarán los nombres de las columnas
for row in table.find_all('th'): 
    heading_table.append(row.text) 
print(heading_table)

content = []
for row in table.find_all('tr'): #recorre todas las filas de la tabla
    if not row.find_all('th'):# ignorar los encabezados
        content.append([element.text for element in row.find_all('td')])
print(content)

shipwrecks = pd.DataFrame(content, columns=heading_table) #volver dataframe
shipwrecks.head()

#------------Ejecicios de BeautifulSoup--------------
import requests 
from bs4 import BeautifulSoup  
URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL) 
soup=BeautifulSoup(req.text, 'lxml')

name_products = []  # Lista donde se almacenan los nombres de los productos
for product in soup.find_all('p', attrs={"class": "t754__title t-name t-name_md js-product-name"}):
    name_products.append(product.text)
print(name_products)

price = []  # Lista donde se almacenan los precios de los productos
for row in soup.find_all('p', attrs={'class': 't754__price-value js-product-price'}):
    price.append(row.text)
print(price)

products_data = (pd.DataFrame())
products_data['name'] = name_products
products_data['price'] = price
print(products_data.head(5))

#------------Solicitudes con parametros--------------
city = 'Lisbon'
URL = f'https://wttr.in/{city}'
PARAM={"format": 4, "m": ""}
req = requests.get(url = URL, params = PARAM)

#------------JSON 2.0--------------
import json #libreria 

x = '{"name": "General Slocum", "date": "June 15, 1904"}'
y = json.loads(x) #convertir JSON a diccionario
print('Name : {0}, date : {1}'.format(y['name'], y['date']))

x = '[{"name": "General Slocum", "date": "June 15, 1904"}, {"name": "Camorta", "date": "May 6, 1902"}]'
y = json.loads(x)
for i in y:
    print('Name : {0}, date : {1}'.format(i['name'], i['date']))

out = json.dumps(y) #convierte los datos de Python al formato JSON
print(out)

lat = 48.8566 # Coordenadas de París
lon = 2.3522
BASE_URL = 'https://api.met.no/weatherapi/locationforecast/2.0/compact'
params = {'lat': lat,'lon': lon}
headers = {
'User-Agent': 'MiAplicacionDePronostico/1.0 (mi_email@ejemplo.com)'
}
response = requests.get(BASE_URL, params=params, headers=headers)
response_parsed = json.loads(response.text) #volver a diccionario

first_forecast = response_parsed['properties']['timeseries'][0]
print(first_forecast)

#------------SQL-------------
Si la caja “Empleados” es una entidad, el perfil del Sr. Peterson será su objeto.
Las columnas se llaman campos. Contienen las características del objeto 
Las filas de la tabla se llaman registros. Cada fila contiene información acerca de un objeto en particular
Una celda es una unidad donde se cruzan un registro y un campo

#sintaxis:
--un comentario de una línea en SQL

/* un comentario de varias líneas
tiene
varias 
línea */

cada declaración (o consulta) termina con un punto y coma ;

SELECT
    name,
    author
FROM
    books
WHERE
    author = 'Stephen King'
    AND column_4 = value_3;

SELECT
    *
FROM
    books;

#------------SQL operadores lógicos-------------
# AND	selecciona las filas para las que se cumplen ambas condiciones
SELECT
    *
FROM
    table_name
WHERE
    condition_1
    AND condition_2;

# OR	selecciona las filas para las que se cumplen una o ambas condiciones
SELECT
    *
FROM
    table_name
WHERE
    condition_1
    OR condition_2;

# NOT	selecciona las filas para las que la condición es falsa
SELECT
    *
FROM
    table_name
WHERE
    condition_1
    AND NOT condition_2;

#------------SQL Rangos-------------
SELECT
    name,
    author,
    date_pub,
    pages
FROM
    books
WHERE
    date_pub > '1995-12-31'
    AND date_pub < '2001-01-01';

SELECT
    name,
    author,
    date_pub,
    pages
FROM
    books
WHERE
    date_pub BETWEEN '1996-01-01'
    AND '2000-12-31';

#------------SQL In-------------
SELECT
    name,
    genre
FROM
    books
WHERE
    genre IN ('Humor', 'Fantasy', 'Young Adult');
#Poner NOT delante del operador IN te permite seleccionar todos los libros cuyos géneros no son los de la  lista

#------------SQL Ejercicios-------------
SELECT name, price, name_store, date_upd 
FROM products_data_all 
WHERE category = 'milk' and date_upd = '2019-06-01';

SELECT
    name,
    price,
    name_store,
    date_upd
FROM
    products_data_all
WHERE
    category = 'milk'
    AND date_upd IN ('2019-06-08', '2019-06-15', '2019-06-22', '2019-06-29');

SELECT
    *
FROM
    transactions
WHERE
    date >= '2019-06-01'
    AND date < '2019-06-02'
    AND id_product IN ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
156, 157, 158, 159, 160, 161);

#------------SQL funciones de agregación-------------
SELECT
    COUNT(*) AS cnt     #devuelve el número de filas
FROM
    books;
WHERE
    author = 'Dean Koontz'; #filtrado que puede ser opcional

# COUNT(*) devuelve el número total de filas de la tabla
# COUNT(column) devuelve el número de valores en una columna
# COUNT(DISTINCT column) devuelve el número de valores únicos en una columna

SELECT
    COUNT(*) AS cnt,
    COUNT(publisher_id) AS publisher_id_cnt,
    COUNT(DISTINCT publisher_id) AS publisher_id_uniq_cnt
FROM
    books;
#los valores resultantes pueden deberse a nulos, repetidos, etc.

SELECT
    SUM(pages) AS total_pages    #Ignora los valores ausentes, sólo funciona con valores numéricos
FROM
    books;

SELECT
    AVG(rating) AS average      #devuelve el valor promedio
FROM
    books;

SELECT 
	MAX(price) as max_price
FROM
    products_data_all;


#------------SQL Ejercicios de funciones de agregación-------------
SELECT
    AVG(price) AS average
FROM
    products_data_all
WHERE
    name = 'Borden Whole Milk, 1 gal'
    AND name_store = 'Wise Penny';

SELECT 
	MAX(price) - MIN(price) as max_min_diff
FROM
    products_data_all
WHERE
    name = 'Meyenberg Goat Milk, 1/2 gal'
    AND name_store = 'Milk Market';

#------------SQL Cambiar tipos de datos-------------
CAST (column_name AS data_type)
column_name :: data_type

SELECT
    AVG(rating::real) AS average
FROM
    books;

SELECT 
	AVG(CAST(weight AS real)) AS average
FROM
    products_data_all
WHERE 
    units = 'oz';

SELECT 
    MAX(CAST(date AS date)) as max_date,
    MIN(CAST(date AS date)) as min_date
FROM
    transactions;

#------------SQL GROUP BY-------------
# se utiliza cuando es necesario dividir los datos en grupos según los valores de los campos
# Una vez que sabes por qué campos agruparás, asegúrate de que todos esos campos estén enumerados tanto en el bloque SELECT como en el bloque GROUP BY.
SELECT
    author,
    genre,
    COUNT(name) AS cnt
FROM
    books
GROUP BY
    author,
    genre;

SELECT
    author,
    AVG(pages) AS avg_pages,
    MAX(pages) AS max_pages
FROM
    books
GROUP BY
    author;

SELECT
    name_store,
    COUNT(name) AS name_cnt,
    COUNT(DISTINCT name) AS name_uniq_cnt
FROM
    products_data_all
GROUP BY
    name_store;

#------------SQL GROUP BY Ejercicios-------------
SELECT 
    category,
	MAX(CAST(weight  AS real)) as max_weight
FROM
    products_data_all
GROUP BY
    category;

SELECT
    name,
    MAX(price) - MIN(price) AS max_min_diff
FROM
    products_data_all
WHERE
    category = 'milk'
    AND date_upd::date = '2019-06-10'
GROUP BY
    name;

#------------SQL ORDER  BY -------------
# Para ordenar los datos por un campo
SELECT
    author,
    COUNT(name) AS cnt
FROM
    books
GROUP BY
    author
ORDER BY
    cnt;    #orden ascendente por defecto
    cnt DESC;       #orden descendente
LIMIT 3;    #limita el número de filas devueltas

#------------SQL ORDER  BY Ejercicios-------------
SELECT
    date_upd::date AS update_date,
    category,
    COUNT(name) AS name_cnt
FROM
    products_data_all
WHERE
    date_upd::date = '2019-06-05'
GROUP BY
    update_date,
    category
ORDER BY
    name_cnt;

SELECT
    date_upd::date AS update_date,
    name_store,
    category,
    COUNT(DISTINCT name) AS uniq_name_cnt
FROM
    products_data_all
WHERE
    name_store = 'T-E-B'
    AND date_upd::date = '2019-06-30'
GROUP BY
    update_date,
    name_store,
    category
ORDER BY
    uniq_name_cnt DESC;

SELECT 
	name,
    MAX(price) as max_price
FROM
    products_data_all
GROUP BY 
    name
ORDER BY 
    max_price DESC
LIMIT 5;

#------------SQL HAVING  BY -------------
La selección resultante incluirá solo aquellas filas para las que la función de agregación produce resultados que cumplen la condición indicada en el bloque HAVING.
WHERE se compila antes de que se realicen las agrupaciones y agregaciones. Por eso es imposible establecer parámetros de filtración que utilicen los resultados de una función de agregación

SELECT
    author,
    COUNT(name) AS name_cnt
FROM
    books
GROUP BY
    author
HAVING
    COUNT(name) > 1
ORDER BY
    name_cnt DESC;

#------------SQL HAVING  BY Ejercicios-------------
SELECT 
    name,
	MAX(price) as max_price
FROM
    products_data_all
GROUP BY
	name
HAVING
    MAX(price)  > 10;

SELECT
    date_upd::date AS update_date,
    name_store,
    COUNT(name) AS name_cnt
FROM
    products_data_all
WHERE
    date_upd::date = '2019-06-03'
    AND units = 'oz'
    AND weight::real > 5
GROUP BY
    update_date,
    name_store
HAVING
    COUNT(name) < 20;

SELECT
    name_store,
	COUNT(DISTINCT name) as name_uniq_cnt
FROM
    products_data_all
GROUP BY
    name_store
HAVING
    COUNT(DISTINCT name) > 30
ORDER BY
    name_uniq_cnt
LIMIT 3;

#------------SQL EXTRACT y DATE_TRUNC -------------
SELECT
    id_user,
    EXTRACT(MONTH FROM log_on) AS month_activity,
    EXTRACT(DAY FROM log_on) AS day_activity
FROM
    user_activity;

# century: siglo
# day: día
# doy: día del año, del 1 al 365/366
# isodow: día de la semana según ISO 8601, el formato internacional de fecha y hora; El lunes es 1, el domingo es 7
# hour: hora
# milliseconds: milisegundos
# minute: minutos
# second: segundos
# month: mes
# quarter: trimestre
# week: semana del año
# year: año

DATE_TRUNC en la fecha truncada resultante se proporciona como un string

'microseconds': microsegundos
milliseconds: milisegundos
second: segundos
minute: minutos
hour: hora
'day': día
week: semana del año
month: mes
quarter: trimestre
'year': año
'decade': década
century: siglo

SELECT
    DATE_TRUNC('hour', log_on) AS date_log_on
FROM
    user_activity;

#------------SQL EXTRACT y DATE_TRUNC Ejercicios-------------
SELECT 
    COUNT(id_product) as cnt,
    EXTRACT(HOUR FROM date) AS hours
FROM
    transactions
GROUP BY
    hours
ORDER BY
    hours

SELECT 
    COUNT(id_product) as cnt,
    DATE_TRUNC('day', date) AS date_day
FROM
    transactions
GROUP BY
    date_day
ORDER BY
    date_day

#------------SQL Subconsultas-------------
SELECT 
    AVG(Sub.count_rating) AS avg_count_rating
FROM
    (SELECT 
        COUNT(rating) as count_rating
    FROM 
        books
    GROUP BY genre) AS Sub;

SELECT 
    name,
    publisher_id
FROM 
    books
WHERE 
    publisher_id = 
            (SELECT 
                 publisher_id
            FROM 
                publisher
            WHERE 
                name ='Knopf');

SELECT 
    name,
    publisher_id
FROM 
    books
WHERE 
    publisher_id IN 
            (SELECT 
                 publisher_id
            FROM 
                publisher
            WHERE 
                name IN ('Knopf', 'Collins', 'Crown'));

#------------SQL Subconsultas |-------------
SELECT
    id_product
FROM
    products_data_all
WHERE (category = 'milk'
    AND price > 17)
    OR (category = 'butter'
        AND price > 9);

SELECT DISTINCT
    user_id
FROM
    transactions
WHERE
    id_product IN (
        SELECT
            id_product
        FROM
            products_data_all
        WHERE (category = 'milk'
            AND price > 17)
        OR (category = 'butter'
            AND price > 9));











































#------------ SQL funciones de ventana -------------
una ventana es un conjunto de filas relacionadas sobre las que se realiza una operación. Este conjunto puede ser la tabla completa o un subconjunto
la función opera en esta "ventana" y devuelve un resultado para cada fila , agregando una nueva columna a tu resultado final.

SELECT
    author_id,
    name,
    price / SUM(price) OVER () AS ratio
FROM
    books_price;

#------------ SQL PARTITION BY -------------
es como un GROUP BY para funciones de ventana.
Divide el conjunto de datos en grupos lógicos (particiones) basados en una columna, y luego aplica el cálculo de la ventana de forma independiente en cada grupo

SELECT
    author_id,
    name,
    price / SUM(price) OVER (PARTITION BY author_id) AS ratio
FROM
    books_price;

#------------ SQL Ejercicios de funciones de ventana y PARTITION BY -------------
SELECT
    name,
    name_store,
    category,
    price,
    price/ AVG(price) OVER (PARTITION BY category, name_store) AS ratio_to_avg_price
FROM
    products_data_all;












SELECT #número acumulado de precios para cada autor
    author_id,
    name,
    SUM(price) OVER (ORDER BY author_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
FROM
    books_price;

SELECT #número acumulado de páginas para cada autor
    author_id,
    name,
    pages,
    SUM(pages) OVER (PARTITION BY author_id ORDER BY author_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
FROM
    books_price;

ORDER BY nos permite definir el orden de ordenación de las filas a través de las cuales se ejecutará la ventana. 

ROWS indicamos los marcos de ventana sobre los que se va a calcular una función de agregación.
UNBOUNDED PRECEDING: todas las filas que están por encima de la actual
N PRECEDING: las n filas por encima de la actual
CURRENT ROW: la fila actual
N FOLLOWING: las n filas debajo de la actual
UNBOUNDED FOLLOWING: todas las filas debajo de la actual

#------------Funciones de clasificación: RANK------------------------------
SELECT #clasificar los libros según el número de páginas de cada autor
    author_id,
    name,
    pages,
    RANK() OVER (PARTITION BY author_id ORDER BY pages)
FROM
    books_price;

devuelve el número de índice de fila en la ventana actual. Si varias filas tienen un valor asignado según las reglas de ORDER BY se les asignará el mismo número

#------------Función de categorización: NTILE------------------------------
Con NTILE podemos poner la fila de salida en un grupo.
El número de grupos en los que se van a dividir los datos se pasa a la función.

SELECT #dividir los libros en cinco categorías según el precio.
    author_id,
    name,
    price,
    ntile(5) OVER (ORDER BY price)
FROM
    books_price;

#------------Funciones de desplazamiento: LAG y LEAD-----------------------------
Puedes pasar a la función el nombre del campo y el desplazamiento (el número de filas) sobre el que se tomará el valor. 
Si no indicas el desplazamiento, será el valor predeterminado: 1.

SELECT #cuántas páginas tiene cada libro y el libro previo del mismo autor
    author_id,
    name,
    pages,
    LAG(pages) OVER (PARTITION BY author_id ORDER BY date_pub)
FROM
    books_price;


#------------Ejercicio-----------------------------
SELECT 
    name_store AS store_name,
    category,
    name as product_name,
    price,
    SUM(price) OVER (PARTITION BY category ORDER BY id_product) AS category_accum, #acumulado por categoria
    SUM(price) OVER (ORDER BY id_product) AS store_accum #acumulado por tienda
FROM
    products_data_all
WHERE
    date_upd::date = '2019-06-02'
    AND name_store = 'Four'
ORDER BY
    id_product;

SELECT DISTINCT
    name_store AS store_name,
    category,
    date_upd::date AS SALE_DATE,
    name,
    price,
    RANK() OVER (PARTITION BY name_store, category ORDER BY price ) as rank #clasificación por tienda y categoría ordenada por precio
FROM
    products_data_all
where
    date_upd::date = '2019-06-02'
order by
    name_store,
    category,
    rank

#------------Nulos en sql ------------------------------
SELECT
    *
FROM
    table_name
WHERE
    column_name IS NULL; #mostrar solo las filas con valores nulos

SELECT
    *
FROM
    table_name
WHERE
    column_name IS NOT NULL; #mostrar solo las filas sin valores nulos

#------------if-elif-else en sql------------------------------
CASE WHEN condition_1 THEN
    result_1
WHEN condition_2 THEN
    result_2
WHEN condition_3 THEN
    result_3
ELSE
    result_4
END;

SELECT
    name,
    CASE WHEN publisher_id IS NULL THEN -1 
    ELSE publisher_id END AS publisher_id_full
FROM
    books;

#------------Ejercicios de los nulos-----------------------------
SELECT
    COUNT(*)
FROM
    products
WHERE
    weight IS NULL; 

SELECT
    AVG(weight::real) as avg_weight,
    units
FROM
    products
GROUP BY
    units;

SELECT
    name,
    CASE WHEN weight IS NULL AND units = 'oz' THEN '23.0705263269575' 
    WHEN weight IS NULL AND units = 'ct' THEN '10.0'
    WHEN weight IS NULL AND units = 'pk' THEN '12.0909090909091'
    WHEN weight IS NULL AND units = 'gal' THEN '0.650793650793651'
    WHEN weight IS NULL AND units = '%' THEN '1.0'
    WHEN weight IS NULL AND units = 'pt' THEN '1.0'
    WHEN weight IS NULL AND units = 'qt' THEN '1.0'
    ELSE weight
    END AS weight_full
FROM
    products;

#------------Buscar substrings ------------------------------
column_name LIKE 'expresión regular'
'%Vampire%' #cualquier cadena que contenga la palabra Vampire

el símbolo % representa cualquier número de caracteres

SELECT
    *
FROM
    books
WHERE
    name LIKE '%Vampire%';

SELECT
    *
FROM
    books
WHERE
    name NOT LIKE '%Vampire%';

column_name LIKE '%!%%' ESCAPE '!'
es un carácter de escape que indica que el siguiente carácter debe interpretarse literalmente

SELECT
    *
FROM
    products
where
    units LIKE '%!%%' ESCAPE '!';

SELECT
    *
FROM
    products
where
    name LIKE '%Moo%';

#------------JOIN-----------------------------
INNER JOIN devuelve solo aquellas filas que tienen valores coincidentes de una tabla a otra (la intersección de las tablas).
OUTER JOIN recupera todos los datos de una tabla y agrega datos de la otra cuando hay filas coincidentes. Hay dos tipos de OUTER JOIN, left (izquierda ) y right (derecha)

#------------INNER JOIN-----------------------------
SELECT 
    TABLE_1.field_1 AS field_1,
    TABLE_1.field_2 AS field_2,
    ...
    TABLE_2.field_n AS field_n
FROM
    TABLE_1
    INNER JOIN TABLE_2 ON TABLE_2.field_1 = TABLE_1.field_2;    #la condición es que field_1 de la segunda tabla coincida con field_2 de la primera.

SELECT
    books.name AS name,
    books.author_id AS books_author_id,
    author.author_id AS author_id, #este se quita en la siguiente para evitar repeticiones
    author.first_name AS first_name,
    author.last_name AS last_name
FROM
    books
    INNER JOIN author ON author.author_id = books.author_id
LIMIT 3;

SELECT
    books.name AS name,
    books.author_id AS books_author_id,
    author.first_name AS first_name,
    author.last_name AS last_name
FROM
    books
    INNER JOIN author ON author.author_id = books.author_id
LIMIT 3;

SELECT
    books.name AS name,
    author.first_name AS first_name,
    author.last_name AS last_name
FROM
    books
    INNER JOIN author ON author.author_id = books.author_id
WHERE
    author.first_name = 'Dean'
    AND author.last_name = 'Koontz';

#------------Ejercicios de inner join-----------------------------
SELECT 
    transactions.id_transaction AS id_transaction,
    products.category AS category,
    products.name as name
FROM
    transactions
    INNER JOIN products ON products.id_product = transactions.id_product 
ORDER BY 
id_transaction
LIMIT 	10;

SELECT DISTINCT     #elimina las filas que son exactamente iguales, manteniendo solo una copia de cada combinación única de valores.
    transactions.date AS date,
    weather.temp as temp,
    weather.rain as rain,
    transactions.id_transaction as id_transaction
FROM
    transactions
INNER JOIN weather ON CAST(weather.date AS date) = CAST(transactions.date AS date)
ORDER BY 
	date DESC;

SELECT distinct 
	products.name
FROM
    products
    INNER JOIN products_stores ON products_stores.id_product = products.id_product
WHERE 
    products_stores.price > 5

SELECT 
	transactions.date as date,
    transactions.id_transaction as id_transaction,
    products.category as category,
    products.name  as name
FROM
    transactions
    INNER JOIN products on transactions.id_product = products.id_product
WHERE 
	products.category = 'butter'
    and transactions.date::date = '2019-06-20'

SELECT 
    products.name as name,
    products.category as category,
    products.units as units,
    products.weight as weight,
    products_stores.price as price
FROM
	products
INNER JOIN products_stores on products_stores.id_product = products.id_product
WHERE 
	products.units = 'oz'
    and products_stores.date_upd ::date = '2019-06-13'

#------------OUTER JOIN: left ------------------------------
LEFT JOIN seleccionará todos los datos de la tabla de la izquierda junto con las filas de la tabla de la derecha que cumplen con la condición de unión. 
RIGHT JOIN hará lo mismo, pero para la tabla de la derecha.
SELECT
    TABLE_1.field_1 AS field_1,
    TABLE_1.field_2 AS field_2,
    ...
    TABLE_2.field_n AS field_n
FROM
    TABLE_1
    LEFT JOIN TABLE_2 ON TABLE_2.field = TABLE_1.field;

SELECT
    author.first_name AS first_name,
    author.last_name AS last_name,
    author.author_id AS author_id,
    books.name AS name,
    books.author_id AS books_author_id
FROM
    author
    LEFT JOIN books ON books.author_id = author.author_id;

FROM    #este seria un right join
    books
    LEFT JOIN author ON books.author_id = author.author_id;

#------------OUTER JOIN: left  Ejercicios------------------------------
SELECT distinct
    products.id_product  as id_product ,
    products.name  as name ,
    products_stores.id_store  as id_store 
FROM
    products
    LEFT JOIN products_stores ON products_stores.id_product = products.id_product
WHERE
	products_stores.id_store is null;

SELECT distinct
	products.name as name
FROM
    products
    LEFT JOIN ( SELECT distinct 
    id_product 
        FROM
            transactions
        WHERE
            id_store = 3
            ) AS subquery ON subquery.id_product = products.id_product
WHERE
    subquery.id_product IS NULL;

SELECT distinct
	products.name as name
FROM
    products
    LEFT JOIN (
    SELECT distinct
    transactions.id_product as id_product,
    transactions.id_store as id_store
        FROM
            transactions
        WHERE
transactions.date ::date = '2019-06-11'
            ) AS subquery ON subquery.id_product = products.id_product
where
    subquery.id_product IS NULL;

#------------OUTER JOIN: right ------------------------------
SELECT
    TABLE_1.field_1 AS field_1,
    TABLE_1.field_2 AS field_2,
    ...
    TABLE_2.field_n AS field_n
FROM
    TABLE_1
    RIGHT JOIN TABLE_2 ON TABLE_1.field = TABLE_2.field;

SELECT
    books.name AS name,
    genre.name AS genre_name
FROM
    books
    RIGHT JOIN genre ON genre.genre_id = books.genre_id;

    RIGHT JOIN books ON books.genre_id = genre.genre_id;

#------------OUTER JOIN: right Ejercicios------------------------------
SELECT 
	CAST(weather.date AS date)
FROM
    transactions
    RIGHT JOIN weather on CAST(weather.date AS date) = CAST(transactions.date AS date)
WHERE 
    transactions.date is null;

SELECT
     products.name as name
FROM ( SELECT DISTINCT 
		id_product 
    FROM
        transactions		
    WHERE
		transactions.id_store = 3
        ) AS subquery
    RIGHT JOIN products ON products.id_product = subquery.id_product
WHERE
    subquery.id_product IS NULL;

#------------Unir varias tablas------------------------------
SELECT 
    TABLE_1.field_1 AS field_1,
    TABLE_1.field_2 AS field_2,
    ...
    TABLE_3.field_n AS field_n
FROM
    TABLE_1
    INNER JOIN TABLE_2 ON TABLE_2.field = TABLE_1.field
    INNER JOIN TABLE_3 ON TABLE_3.field = TABLE_1.field;

SELECT
    books.name AS books_name,
    books.author_id AS books_author_id,
    author.first_name AS author_first_name,
    author.last_name AS author_last_name,
    books.genre_id AS books_genre_id,
    genre.name AS genre_name
FROM
    books
    INNER JOIN author ON author.author_id = books.author_id
    INNER JOIN genre ON genre.genre_id = books.genre_id;

SELECT
    books.name AS name,
    genre.name AS genre_name,
    author.first_name AS first_name,
    author.last_name AS last_name
FROM
    books
    INNER JOIN author ON author.author_id = books.author_id
    RIGHT JOIN genre ON genre.genre_id = books.genre_id;

SELECT 
	transactions.id_transaction as id_transaction,
    stores.name_store as name_store, 
    products.category as category,
    products.name as name 
FROM
    transactions
    INNER JOIN products on transactions.id_product = products.id_product
    INNER JOIN stores on transactions.id_store = stores.id_store
WHERE 
    transactions.date::date = '2019-06-05'

SELECT
    CAST(weather.date AS date) AS date,
    weather.temp  as temp ,
    weather.rain  as rain ,
    products.name  as name 
FROM
    weather
    LEFT JOIN transactions ON CAST(transactions.date AS date) = CAST(weather.date AS date)
    LEFT JOIN products ON products.id_product = transactions.id_product
ORDER BY 
    date DESC,
    name
LIMIT 30;

SELECT 
	transactions.id_transaction  as id_transaction ,
    products.name as name
FROM
    transactions
    INNER JOIN products ON products.id_product = transactions.id_product
    INNER JOIN weather ON CAST(transactions.date AS date) = CAST(weather.date AS date)
WHERE 
    weather.rain = False

#------------Funciones de agregacion con join------------------------------
SELECT
    genre.name AS genre_name,
    COUNT(books.name) AS name_cnt
FROM
    books
    INNER JOIN genre ON genre.genre_id = books.genre_id
GROUP BY
    genre_name;

SELECT
    genre.name AS genre_name,
    author.first_name AS author_first_name,
    author.last_name AS author_last_name,
    COUNT(books.name) AS name_cnt
FROM
    books
    INNER JOIN genre ON genre.genre_id = books.genre_id
    INNER JOIN author ON author.author_id = books.author_id
GROUP BY
    genre_name,
    author_first_name,
    author_last_name;

SELECT
FROM
    table_1
    LEFT JOIN table_2 ON condition 1
        AND condition 2
        AND condition 3

SELECT 
    transactions.id_transaction as id_transaction,
	count(products.name) as name_cnt,
    COUNT(DISTINCT products.name)as name_uniq_cnt 
FROM
    transactions
    INNER JOIN products on transactions.id_product = products.id_product
GROUP BY 
    id_transaction
LIMIT 10;

SELECT
    transactions.id_transaction as id_transaction,
	count(products.name) as name_cnt,
    COUNT(DISTINCT products.name)as name_uniq_cnt 
FROM
    transactions
    INNER JOIN products on transactions.id_product = products.id_product
GROUP BY
transactions.id_transaction
HAVING
    COUNT(products.name) != COUNT(DISTINCT products.name)

SELECT 
    weather.rain as rain,
    COUNT(DISTINCT transactions.id_transaction )as uniq_transactions 
FROM
    transactions
    INNER JOIN weather on CAST(weather.date as date) = CAST(transactions.date as date)
GROUP BY 
    weather.rain 

SELECT 
	CAST(weather.date as date),
    weather.temp as temp,
    COUNT(DISTINCT transactions.id_transaction )as uniq_transactions 
FROM
    weather
    LEFT JOIN transactions on CAST(weather.date as date) = CAST(transactions.date as date)
GROUP BY 
    weather.date,
	weather.temp
ORDER BY 
    weather.date

SELECT 
	transactions.id_transaction as id_transaction,
    SUM(products_stores.price) as total,
    COUNT(products_stores.id_product) as amount
FROM
    transactions
    LEFT JOIN products_stores on CAST(products_stores.date_upd AS date) = CAST(transactions.date AS date) and products_stores.id_product = transactions.id_product and products_stores.id_store = transactions.id_store 
GROUP BY 
transactions.id_transaction 
having 
SUM(products_stores.price) > 35

SELECT distinct
    products.name as name
from 
    products 
    LEFT JOIN (
        SELECT
            transactions.id_product
        FROM
            transactions
        WHERE
        transactions.date::date = '2019-06-01'
) AS subq ON products.id_product = subq.id_product
WHERE
    subq.id_product IS NOT NULL;

#------------unir los datos recuperados por consultas separadas------------------------------
UNION y UNION ALL se utilizan para unir los datos de tablas.

SELECT
    column_name_1
FROM
    table_1
UNION #o UNION ALL
SELECT
    column_name_1
FROM
    table_2;

condiciones:
La primera y la segunda tabla deben coincidir por el número de columnas seleccionadas y sus tipos de datos
Los campos deben estar en el mismo orden en la primera y segunda tabla.

SELECT
    year_tag,
    name,
    weight
FROM
    cows_grandpa
UNION
SELECT
    tag,
    name,
    weight
FROM
    cows_grandma;

UNION excluye duplicados en la selección resultante.
UNION ALL incluye duplicados en la selección resultante.

SELECT
    year_tag,
    name,
    weight
FROM
    cow
UNION ALL
SELECT
    year_tag,
    name,
    weight
FROM
    bulls;

SELECT DISTINCT
    products.name AS name
FROM
    products
    LEFT JOIN (
        SELECT
            id_product
        FROM
            transactions
        WHERE
            CAST(transactions.date AS date) = '2019-06-01') AS SUBQ1 ON products.id_product = SUBQ1.id_product
WHERE
    SUBQ1.id_product IS NOT NULL
    union
SELECT DISTINCT
    products.name AS name
FROM
    products
    LEFT JOIN (
        SELECT
            id_product
        FROM
            transactions
        WHERE
            CAST(transactions.date AS date) = '2019-06-08') AS SUBQ2 ON products.id_product = SUBQ2.id_product
WHERE
    SUBQ2.id_product IS NOT NULL;

select 
    count(name)
from
( SELECT DISTINCT
        products.name AS name
    FROM
        products
    LEFT JOIN (
        SELECT
            id_product
        FROM
            transactions
        WHERE
            CAST(transactions.date AS date) = '2019-06-01') AS SUBQ1 ON products.id_product = SUBQ1.id_product
    WHERE
        SUBQ1.id_product IS NOT NULL
    UNION
    SELECT DISTINCT
        products.name AS name
    FROM
        products
    LEFT JOIN (
        SELECT
            id_product
        FROM
            transactions
        WHERE
            CAST(transactions.date AS date) = '2019-06-08') AS SUBQ2 ON products.id_product = SUBQ2.id_product
    WHERE
        SUBQ2.id_product IS NOT NULL) as SUBQ

select
    count(name)
from
( SELECT DISTINCT
        products.name AS name
    FROM
        products
    LEFT JOIN (
        SELECT
            id_product
        FROM
            transactions
        WHERE
            CAST(transactions.date AS date) = '2019-06-01') AS SUBQ1 ON products.id_product = SUBQ1.id_product
    WHERE
        SUBQ1.id_product IS NOT NULL
        union all
        SELECT DISTINCT
            products.name AS name
        FROM
            products
        LEFT JOIN (
            SELECT
                id_product
            FROM
                transactions
            WHERE
                CAST(transactions.date AS date) = '2019-06-08') AS SUBQ2 ON products.id_product = SUBQ2.id_product
        WHERE
            SUBQ2.id_product IS NOT NULL)
    as SUBQ

#------------Crear dataframe con informacion de internet------------------------------
import requests, pandas as pd
from bs4 import BeautifulSoup 
import json

URL = 'https://practicum-content.s3.us-west-1.amazonaws.com/data-analyst-eng/moved_chicago_weather_2017.html'
response = requests.get(URL)
soup=BeautifulSoup(response.text, 'lxml')
table = soup.find('table', attrs={"id": "weather_records"})

heading_table = []
for row in table.find_all('th'):
    heading_table.append(row.text)

content = []
for row in table.find_all('tr'):
    if not row.find_all('th'):  # Solo filas sin encabezados
        content.append([element.text for element in row.find_all('td')])

weather_records = pd.DataFrame(content, columns=heading_table)
print(weather_records)