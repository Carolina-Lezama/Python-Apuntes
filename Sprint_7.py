la interfaz de línea de comandos (CLI)
interfaces gráficas de usuario (GUI)

Presiona Win + R para abrir la ventana Ejecutar

cada vez que quieras abrir Linux desde Windows, deberás dirigirte a PowerShell y escribir wsl. Mientras mantengas esta ventana abierta, Linux estará corriendo en paralelo en tu computadora junto a Windows.
Ctrl + Alt + T o buscando "terminal" en el Dash.

comando -opciones del comando -argumentos del comando

las opciones modifican el comportamiento del comando y se indican con un guion (-) o dos guiones (--) antes del nombre de la opción.
si no se dan se usaran valores predeterminados
puedes incluir una o varias opciones después del comando

los argumentos son los elementos sobre los que actúa el comando y se escriben después del comando y sus opciones.
también pueden aceptar uno o varios argumentos, que son datos adicionales sobre los que el comando actuará 

ls -r   muestra los archivos en orden inverso

ls -r -a
ls -ra  muestra todos los archivos, incluidos los ocultos, en orden inverso

los nombres de todos los archivos y directorios ocultos comienzan siempre por .

ls -r sql_project   muestra los archivos y directorios dentro del directorio sql_project en orden inverso

ls --help   estás pidiendo al sistema que te muestre un mensaje de ayuda detallado sobre ese comando en particular.
man ls      abre el manual de usuario del comando ls

pwd  responde a las siglas en ingles "print working directory"
el comando pwd te muestra la ruta completa del directorio en el que te encuentras actualmente en la línea de comandos.

Directorio actual (./)
cd ./intro_to_ml/

cd ~ nos llevaría directamente a /home/anthony/ (Tu carpeta personal )

clear no borra el historial de los comandos que has ejecutado. Simplemente limpia la visualización en la pantalla.
Importante: Ten en cuenta que este historial de comandos se guarda solo mientras la ventana de la Terminal esté abierta.

Si presionas la tecla Tabulador dos veces rápidamente, la Terminal te mostrará una lista de todos los comandos que comienzan con "pw":

mkdir mi_nueva_carpeta      crea un nuevo directorio  en el directorio actual.
mkdir carpeta_uno carpeta_dos carpeta_tres     multiples directorios
mkdir mi_carpeta_principal/mi_subcarpeta       Creando carpetas dentro de carpetas (subdirectorios); olo funcionará si la carpeta principal ya existe
mkdir -p mi_carpeta_principal/mi_subcarpeta         p sirve para crear ambas si ninguna existe

Usar la opción -v le dice al comando: "Por favor, muéstrame en la pantalla cada acción que estás realizando"
mkdir -p -v mi_carpeta_principal/mi_subcarpeta

#-----------Eliminar directorios vacios--------------
# ~$ rm -d mi_carpeta_temporal
# ~$ rm --dir mi_carpeta_temporal

#-----------Pedir confirmacion--------------
# ~$ rm -d -i mi_carpeta_temporal

#-----------Comando echo--------------
# ~$ echo Hola, mundo de la CLI    mostrar texto en la pantalla
# ~$ echo -e 'Quiero aprender sobre la CLI.\n¡Todo programador la conoce!'    -e habilita los caracteres de escape

# echo "El inicio de una gran aventura..." > aventura_parte1.txt
# echo "Y así continuó el viaje." > aventura_parte2.txt
# # ~$ echo "Aprendiendo comandos básicos de la CLI" > mi_primer_archivo.txt    enviar su salida directamente a un archivo de texto
# # Si el archivo.txt no existe, lo creará automáticamente. 
# # Si ya existe, el símbolo > sobrescribirá todo su contenido con el nuevo texto.

# # ~$ echo "¡Este es un nuevo renglón!" >> mi_primer_archivo.txt   agregar nuevo texto al final de un archivo existente sin borrar su contenido

# #-----------Comando cat--------------
# ~$ cat mi_primer_archivo.txt    mostrar el contenido de un archivo de texto en la pantalla
# cat importante.txt
# cat aventura_parte1.txt aventura_parte2.txt

# cat > mi_nota.txt   crear un archivo de texto y escribir en él
# # escribe el texto que quieras y cuando termines, presiona Ctrl + D para guardar y salir

# cat importante.txt > copia_importante.txt   copiar el contenido de un archivo a otro

# #-----------Comando touch--------------
# crea archivos vacíos de forma instantánea. A estos se les llama "archivos dummy" porque existen, pero no contienen ningún dato.
# ~$ touch mi_archivo_vacio.txt

# #-----------Comando mv--------------
# ~$ touch mi_documento.txt   #mover la ubicacion del archivo
# ~$ mkdir mis_documentos
# ~$ mv mi_documento.txt mis_documentos

# ~$ mkdir ~/archivos_importantes
# ~$ mv mi_documento.txt ~/archivos_importantes

# ~$ touch archivo_uno.txt archivo_dos.txt archivo_tres.txt
# ~$ mkdir carpeta_destino
# ~$ mv archivo_uno.txt archivo_dos.txt archivo_tres.txt carpeta_destino

# ~$ mkdir mi_carpeta_antigua mis_archivos    mover carpetas
# ~$ mv mi_carpeta_antigua mis_archivos

# ~$ touch documento_viejo.txt
# ~$ mv documento_viejo.txt documento_nuevo.txt   #cambiar el nombre de un archivo
# al renombrar un archivo a un nombre que ya existe, ya que el archivo existente será reemplazado

# #-----------Comando cp--------------
# ~$ touch informe_original.txt
# ~$ cp informe_original.txt copia_informe.txt    hubicacion y nombre

# ~$ mkdir ~/respaldo_documentos      copiar carpeta con documentos dentro
# ~$ cp -r mis_documentos ~/respaldo_documentos

# #-----------Caracteres comodines--------------
# para representar uno o varios caracteres en un patrón de búsqueda

# El comodín asterisco (*)
# ~$ ls foto_*.jpg
# ~$ rm temp_*
# ~$ ls *.jpg
# coincide con la parte inicial de los nombres de archivo, * coincide con cualquier secuencia de caracteres que haya en medio (como la fecha)
# puede coincidir con cualquier cantidad de caracteres (incluso ninguno)

# El comodín signo de interrogación (?)
# coincide exactamente con un carácter
# ~$ ls foto_?.jpg

# El comodín corchetes ([])
# definir un conjunto de caracteres permitidos en una posición dada
# ~$ ls foto_[123a].jpg
# ~$ ls [A-Z]*
# listar solo las fotos que tienen un único carácter después del guion bajo que sea un dígito del 1 al 3 o la letra 'a'
# [a-z] coincide con cualquier letra minúscula (a hasta z).
# [A-Z] coincide con cualquier letra mayúscula (A hasta Z).
# [0-9] coincide con cualquier dígito (0 hasta 9).

# El comodín llaves ({})
# aplicar un comando a múltiples patrones alternativos
# ~$ ls foto_*.{jpg,png,jpeg}
# ls {file*.txt,e[vw]e.txt}
# ls {file?.txt,eve.txt,ewe.txt}

# #-----------Head y Tail--------------
# #por defecto son 10 lineas
# # ~$ head nombres.txt
# ~$ head -n 4 nombres.txt

# #-----------Contar palabras-------------
# ~$ wc archivo.txt
# 15   120   850 archivo.txt
# 15 → número de líneas
# 120 → número de palabras
# 850 → número de bytes (caracteres)

# #----------Uso de vi-------------
# vi my_new_text_file.txt     crear un nuevo archivo o abrir uno ya existente

# entrar al modo de inserción presionando i
# volver al modo comando usando esc

# comandos predecedidos por :
# :wq guardar y salir.
# ZZ igual que :wq
# :q! para salir del archivo sin guardar los cambios.

# desde modo comando:
# dd para borrar una línea seleccionada.
# x para borrar un carácter seleccionado.
# r para reemplazar un carácter seleccionado.

# #-----------Comando find--------------
# buscar por nombre
# find . -name datos_importantes.csv
# find . -name "*.csv"
# (el punto . representa el directorio actual); esto se puede reemplazar con la ruta completa del directorio donde quieres iniciar la búsqueda

# find . -iname datos_importantes.csv     buscar sin importar mayúsculas o minúsculas

# buscar por tipo
# find . -type d    buscar solo directorios
# find . -type f    buscar solo archivos
# find /home/ -type f -name "log_file_*"

# #-----------Comando grep--------------
permite buscar patrones específicos dentro del contenido de uno o varios archivos de texto.
distingue entre mayúsculas y minúsculas

grep "dato" ejemplo.txt         encontrar todas las líneas que contienen la palabra
grep -w "dato" ejemplo.txt      buscar la palabra exacta
grep -iw "dato" ejemplo.txt     ignorar mayúsculas y minúsculas

grep -iw "dato" *      buscar en todos los archivos de texto del directorio actual
grep -l "datos" ejemplo.txt datos_uno.txt info.txt    mostrar solo los nombres de los archivos que contienen el patrón buscado

# #-----------Ver alias(comandos rapidos personalizados)--------------
# ~$ alias

alias nombre_del_atajo="comando_a_ejecutar"     crear un alias
alias ll="ls -la"       sin espacios 
alias irproyecto="cd /home/tu_usuario/trabajo/mi_proyecto_genial"
alias moverseguro="mv -i"

unalias irproyecto      eliminar un alias

# #-----------Crear alias permanentes--------------
vi ~./bashrc      abrir el archivo de configuración de bash, vi solo es un ejemplo

# mis alias personalizados
alias cdp="cd /home/my_amazing_project"
alias smv="mv -i”

# #-----------Usar alias--------------
lsr /home/usuario      puedes usar lsr como si fuera un comando normal, seguido de los argumentos que el comando original

#-----------Ver variables de entorno--------------
env
printenv

echo $HOME     saber el valor de una variable en particular

MI_VARIABLE="mi primer valor"       crear una variable de entorno temporal
echo $MI_VARIABLE

export MI_VARIABLE     hacer que la variable esté disponible para otros programas y scripts iniciados desde la Terminal actual
export OTRA_VARIABLE="otro valor"
solo durará mientras tu ventana de la Terminal esté abierta

#-----------Importar variables desde pyhton--------------
from dotenv import load_dotenv
import os

archivo .env
RUTA_DATOS=/home/tu_usuario/proyecto/data
API_KEY_SERVICIO_X=abcdef123456
PATH_MODELO_ML=modelos/classificador.pt
UMBRAL_CONFIANZA=0.85

archivo .py
# Carga las variables de entorno desde el archivo .env (si existe)
load_dotenv()

# Ahora puedes acceder a las variables como si estuvieran definidas en el sistema
ruta_datos = os.environ.get('RUTA_DATOS')
api_key_x = os.environ.get('API_KEY_SERVICIO_X')
path_modelo = os.environ.get('PATH_MODELO_ML')
umbral = os.environ.get('UMBRAL_CONFIANZA')

print(f"Ruta a los datos: {ruta_datos}")
print(f"Clave de API del Servicio X (parcial): {api_key_x[:5] if api_key_x else None}")
print(f"Ruta al modelo de ML: {path_modelo}")
print(f"Umbral de confianza: {umbral}")

#-----------Pipes--------------
te permiten conectar la salida de un comando directamente a la entrada de otro

comando1 | comando2 | comando3 | ...
ls | grep "data"
ls | tail -1

#-----------History--------------
history + enter
history 10      ver los últimos 10

!2     re-ejecutar el comando 2
history | grep "ls"     pipear el resultado y luego buscar 
history > mi_historial.txt      guardar historial














#-----------Creando entornos virtuales desde conda--------------
conda create --name mi_entorno       tenemos que estas en la ubicacion donde se creara
conda activate mi_entorno       activar el entorno virtual
conda deactivate      desactivar el entorno virtual

conda install -c plotly plotly     hacer esto dentro del entorno virtual para instalar solo aqui
conda install --name mi_entorno scipy     o sin estar activo

conda create --name mi_entorno scipy     crear e intalar a la vez

conda install --name mi_entorno scipy=0.15.0
conda create --name mi_entorno python=3.9    versiones especificas

conda list --export > requirements.txt      archivo con los nombres de las librerías y sus versiones exactas

conda env create --name mi_nuevo_entorno --file requirements.txt    crear entonro con las especificaciones del .txt
conda create --name customers_analysis python=3.7 pandas=1.5.3
conda install --name customers_analysis pandas=1.5.3 python=3.7
conda create --name customers_analysis python=3.7 y luego conda install pandas=1.5.3

#-----------Comando clon en git--------------
git clone [URL_DEL_REPOSITORIO]     clonar-descargar algun repositorio
git clone https://github.com/Carolina-Lezama/numpy

#-----------Comando pull en git--------------
git pull    actualiza el repositorio local con los cambios del remoto

C:\Users\michi>cd numpy
C:\Users\michi\numpy>git pull

#-----------Comando status en git--------------
git status      ver si hubo cambios en el repositorio

#-----------Comando add en git--------------
git add [archivo]    preparar los archivos al area de stage para el siguiente commit
git add .

C:\Users\michi\Downloads\numpy>git add .

#-----------Comando commit en git--------------
git commit -m "Descripción de los cambios"  guardar los cambios en el repositorio como punto de control
git commit -m "Columna nombre y apellido agregada"

C:\Users\michi\Downloads\numpy>git commit -m "add notes file"

#-----------Comando push en git--------------
git push   subir los cambios a github

#-----------Correr scripts en cmd--------------
python3 image_rotator.py

#-----------Correr scripts con parametros en cmd--------------
py script_imagenes_parametros.py tripleten_logo.png output.png 180  

Las opciones se pueden pasar en cualquier orden (no son posicionales). 
Los argumentos son obligatorios, las opciones no lo son (de ahí su nombre).

#-----------Instalar paquetes--------------
py -m pip install Pillow  

#-----------Opciones en cmd--------------
--option=value, --option value      -o=value -o value  hacen el mismo efecto
Un valor por defecto se especifica con el parámetro default= en add_argument().
parser.add_argument('--angle', '-a', type=int, default=90) # tercer argumento: ángulo

$ python3 image_rotator.py tripleten_logo.jpeg --angle 90 output.png

#----------Flags--------------
son opciones booleanas especiales
debemos incluir el parámetro action= en add_argument(). el flag se establecerá en False por defecto
parser.add_argument('-i', '--info', action='store_true')

python3 image_rotator.py tripleten_logo.jpeg --angle 180 output.png -i      activar la flag

#----------Argumentos posicionales--------------
def trip_price(dist_miles, mpg, price, loc_from, loc_to):
        total_price = dist_miles * price / mpg
    print(f'Un viaje de {loc_from} a {loc_to} costará ${total_price}')

trip_price(409, 35, 5.1, 'A', 'B')

#----------Argumentos de palabras clave--------------
def trip_price(dist_miles, mpg, price, loc_from, loc_to):
    total_price = dist_miles * price / mpg
    print(f'Un viaje de {loc_from} a {loc_to} costará ${total_price}')

trip_price(dist_miles=409, loc_from='A', loc_to='B', mpg=35, price=5.1)

#----------Argumentos predeterminados--------------
def trip_price(dist_miles, mpg, price, loc_from='A', loc_to='B'): #tenemos que poner todos ellos después de los parámetros que no tienen valores predeterminados
    total_price = dist_miles * price / mpg
    print(f'Un viaje de {loc_from} a {loc_to} costará ${total_price}')

trip_price(409, 35, price=3.8)

#----------Tipos en los argumentos--------------
def list_of_words(text: str, sep: str = " ") -> list:
    return text.split(sep)
text debe ser tipo str.el argumento opcional sep también debe ser str (el valor por defecto es " ").  -> list especifica que list_of_words() devolverá una lista.
mismo comportamiento a uno donde no se especifica

#----------Comprobación de tipo--------------
Detecta errores de tipo antes de que tu código se ejecute.
$ mypy list_of_words.py 
revisó nuestro script en busca de incumplimientos, pero no lo ejecutó.

#----------Try-except--------------
try:
    im = Image.open(args.input_file)

except FileNotFoundError:
    print('archivo de entrada no encontrado, introduce el nombre de archivo correcto')

try:
    im = Image.open(args.input_file)

except Exception as error_msg:
    print(error_msg)
    print('the default image is used')
    im = Image.open('default_input.png')

#----------Try-except-else--------------
qué ocurrirá en el caso de que no se encuentre ningún error
try:
    im = Image.open(args.input_file)
    
except FileNotFoundError:
    print('archivo de entrada no encontrado, introduce el nombre de archivo correcto')

else:
    rotated = im.rotate(angle)
    im.close() # cerrar el archivo de imagen, eliminándolo de la memoria
    rotated.save(args.output_file)
    print("Ejecución fluida'")

#----------Instalar paquetes-------------
pip install --user Pillow

#----------importacion------------
import re
def check_email(string: str):
    '''
    usa expresiones regulares para comprobar el formato de la dirección de correo electrónico
    el patrón es "algo@algo.algo"
    '''
    if re.match('[.\w]+@\w+[.]\w+', string):
        print('correcto')
    else:
        print('comprobar dirección de correo electrónico')
def check_age(age: int):
    if age >= 50:
        print('acceso permitido')
    else:
        print("eres demasiado joven")

import module
email = input()
module.check_email(email)

#----------Ejecutar solo en main------------
#Si importamos module_1.py en otro script, la función estará disponible, pero el bloque dentro de if __name__ == '__main__' no se ejecutará
def useful_function():
    print('funcionando')

if __name__ == '__main__':
    print('probando función...')
    useful_function()

#----------Forma de agregar modulos-----------
# paquetes incorporados
import math
import os
# paquetes de terceros
import pandas
import numpy
# mi propio módulo
import mymodule
# el resto del programa

#----------Crear clases-----------
class Knight: # crear la clase Knight
    def __init__(self, name):
        # establecer parámetros
        self.health = 100
        self.damage = 25
        self.knowledge = 20
        self.name = name

arthur = Knight('Arthur')
richard = Knight('Richard')

#----------Acceder a los atributos-----------
print(arthur.health)
print(arthur.damage)
print(arthur.knowledge)
print(arthur.name)

#----------Cambiar los atributos----------
arthur.health = 150
print(arthur.health)

#----------Ver todos los atributos----------
print(arthur.__dict__)

#----------Metodos en clases----------
class Knight:
    def __init__(self, name):
        self.health = 100
        self.damage = 25
        self.knowledge = 20
        self.name = name
    
    def heal(self): #self significa que la llamada a estos métodos afecta a la instancia que los llama
        self.health += 20
    def learn(self):
        self.knowledge += 5 #estatico
        
    def heal2(self, amount):
        self.health += amount #dinamico
    def learn2(self, amount):
        self.knowledge += amount

#----------Crear instancia---------
arthur = Knight('Arthur')

#----------Llamar metodos----------
arthur.heal2(10)
arthur.learn2(2)

arthur.heal()
arthur.learn()

#----------Metodos estaticos----------
# no están vinculados a una instancia específica de una clase ni requieren el parámetro self
# pueden ser llamados sin crear un objeto de esa clase 
# no tienen la capacidad de modificar el estado de un objeto

class Stock:
      def __init__(self, ticker, amount):
            self.ticker = ticker
            self.amount = amount

      @staticmethod
    def show_current_price(ticker):
        current = # Aquí implementarías código
        print(current)

Stock.show_current_price('Apple') #utilizar metodo estatico

#----------Metodos de clase----------
class Stock:
    def __init__(self, ticker, amount, price):
        self.ticker = ticker
        self.amount = amount
        self.price = price
        self.total = self.amount * self.price

    @classmethod
    def from_string(cls, string):
        ticker, amount, price = string.split()
        return cls(ticker, int(amount), float(price))

# Crear instancia desde un string
abc = Stock.from_string('ABC 10 1.5')

#----------Herencia----------
class Character: #clase padre
    def __init__(self, name):
        self.name = name
        self.health = 100
    def heal(self, value:int=20):
        self.health += value
    def learn(self, value:int=20):
        self.knowledge += value

class Knight(Character): #clase hijo
    def __init__(self, name):
        Character.__init__(self, name) # se hereda el constructor de la clase padre
        self.damage = 25 # atributo añadido
        self.knowledge = 20 # atributo añadido

class Peasant(Character):
    def __init__(self, name):
        super().__init__(name) # otra forma de heredar el constructor de la clase padre
        self.damage = 10
        self.knowledge = 36

arthur = Knight('Arthur')
arthur.heal() # se utiliza el valor predeterminado
arthur.learn(100) # valor personalizado
print(arthur.__dict__)

#----------Ejmplo con todo lo anterior----------
import datetime as dt
class Account:
    def __init__(self, bank,    , holder_id, balance:float=0.0):
        self.bank = bank
        self.acc_id = acc_id
        self.holder_id = holder_id
        self.balance = balance
        self.start_date = dt.datetime.now()
    def deposit(self, amount:float):
        self.balance += amount
    def withdraw(self, amount:float):
        self.balance -= amount
    @staticmethod
    def bankphone(bank):
        '''
        imprime el número del banco
        '''
        print('1-000-1234567')
    @classmethod
    def quick(cls, string):
        '''
        crea una cuenta a partir de una cadena
        usando solo las identificaciones de cuenta y titular
        separadas por una barra
        '''
        acc_id, holder_id = string.split('/')
        return cls('default_bank', acc_id, holder_id, 0)
first = Account('old_trusty', '001', '10043', 500)
first.deposit(250)
first.withdraw(400)
print(first.balance)
second = Account.quick('002/10123')
print(second.start_date.year)

#----------Leer desde archivos----------
f = open('my_file.txt')     #los archivos se abren en modo “lectura”
f = open('my_file.txt', mode='r')

#----------Imprimir archivo----------
print(f.read())   #lee todo el archivo 
print(f.readlines())  #lee todas las líneas y las devuelve como una lista de strings

#----------Cerrar archivo----------
f.close()

#----------Administrador de contexto----------
with open('my_file.txt') as f:
    for line in f.readlines():    
        print(line.rstrip()) #cierra automaticamente cuando termina el with

#----------Escritura----------
'w': abre un nuevo archivo vacío, en caso de que ya exista, se sobrescribirá
'a': abre un archivo nuevo o ya existente y le añade texto.

now = datetime.now() 
with open('my_journal.txt', 'a') as f:
    f.write('\n')     # comienza con una nueva línea
      f.write(str(now)) # marca temporal
    f.write('\n\n')   # inserta una línea en blanco
      f.write(' ')      # espacio vacío
      f.write(input())  # escribe una entrada en el diario desde el teclado
      f.write('\n\n')   # termina con una línea en blanco

titles = ['Planeta del futuro\n', 'Soluciones para un mundo sostenible\n', "La ciudad más contaminada del mundo\n"]
with open('output.txt', 'w') as f:
    f.writelines(titles)

#----------Abrir formato JSON----------
import json
import requests # librería para obtener datos de internet
data = requests.get('https://dummyjson.com/products/1') #se descarga como una cadena 
text = data.text
print(json.loads(text)) # convertirlo en un archivo JSON

#----------Escribir formato JSON----------
import json
data = dict(user_id=12, status='active', user_name='Rachel')
with open ('output.json', 'w') as f:
    json.dump(data, f)

#----------Ejemplo----------
import json
import requests

# obtener datos de internet
response = requests.get('https://dummyjson.com/products/category/smartphones')
text = response.text

json_data = json.loads(text)
products = json_data['products'] 

items = []
brand = 'Samsung'

for entry in products:
    if entry['brand'] == brand:
        items.append(entry) 

with open('samsung_items.json', 'w') as f:
    json.dump(items, f)

#----------Ejemplo----------
lines = ['uno', 'dos', 'tres']

with open('lines.txt', 'w') as f:
    f.writelines('\n'.join(lines))
    
with open('lines.txt') as f:
    for line in f.readlines():
        print(line.rstrip())

#----------Request----------
import requests
response = requests.get("https://api.frankfurter.app/latest")
print(response) #vemos el código de estado HTTP por defecto
print(response.json())

import requests
params = {"from":"USD", "to":"GBP", "amount":20} #pasar parametros, el valor por defecto es EUR
res = requests.get("https://api.frankfurter.app/latest", params=params)
print(res.json())
# {'amount': 20.0, 'base': 'USD', 'date': '2025-04-04', 'rates': {'GBP': 15.3722}}

import requests
response = requests.get("https://api.frankfurter.app/latest")
print(response.headers['Server']) #ver las cabeceras de la respuesta
print(response.headers['Content-Type'])
print(response.headers['Content-Encoding'])

#----------Uso de api con apikey----------
import requests
api_key = 'zC5JOYIAeAgn92Z9CJKUQE1ns3dvHeyf'  # este es un ejemplo de la clave API, no es real
params = {'apikey': api_key, 'q': 'New York'}
aw_location_url = "https://dataservice.accuweather.com/locations/v1/cities/search"
    aw_location_res = requests.get(aw_location_url, params=params)

import pprint
pprint.pprint(aw_location_res.json()) # ver la respuesta completa

for loc_info in aw_location_res.json():
    print('{:>8}   {:10}   {:16}    {:16}'.format(
        loc_info['Key'],
        loc_info['EnglishName'],
        loc_info['Country']['EnglishName'],
        loc_info['AdministrativeArea']['EnglishName']))










import requests

api_key = 'zC5JOYIAeAgn92Z9CJKUQE1ns3dvHeyf'
params = {'apikey': api_key, 'metric': True}

location_id = 349727
aw_forecast_url = "https://dataservice.accuweather.com/forecasts/v1/daily/5day/" + str(location_id)
aw_forecast_res = requests.get(aw_forecast_url, params=params)


for daily_forecast in aw_forecast_res.json()['DailyForecasts']:
    print('{}   {:30} {}{}  {}{}'.format(
        daily_forecast['Date'], 
        daily_forecast['Day']['IconPhrase'], 
        daily_forecast['Temperature']['Minimum']['Value'],
        daily_forecast['Temperature']['Minimum']['Unit'], 
        daily_forecast['Temperature']['Maximum']['Value'],
        daily_forecast['Temperature']['Maximum']['Unit']))







import requests

url = 'https://dummyjson.com/products'

params  = {'limit': 3}
response = requests.get(url, params=params)
print(response.json())








import requests

base_url = 'https://collectionapi.metmuseum.org/'
url = base_url + 'public/collection/v1/objects/437133'
re = requests.get(url)
response = re.json()
print(response['artistDisplayName'])



import requests

base_url = 'https://collectionapi.metmuseum.org/'
url = base_url + 'public/collection/v1/departments'

response = requests.get(url)

for dpt in response.json()['departments']:
    if 'Art' in dpt['displayName']:
        print(dpt)
















#----------Uso de api con apikey----------
# streamlit run app.py    correr mi aplicacion del repositorio
# http://localhost:8501/

#----------Crear aplicacion en render----------
# https://dashboard.render.com/web/new
# En la sección Build Command, agregamos pip install --upgrade pip && pip install -r requirements.txt.
# En la sección Start Command, agregamos streamlit run app.py.
# https://experimento-uno.onrender.com/

#----------variables de estado de Streamlit----------
mantener valores sobre nuevas ejecuciones de una aplicación

#----------Pylint----------
Los linters pueden comprobar diferentes aspectos del estilo de código
pylint my_script.py     ejecucion
generará una salida donde el código no sigue el estilo de código, que es PEP 8 por defecto

- ************ Module my_script
my_script.py:6:8: C0303: Trailing whitespace (trailing-whitespace)
my_script.py:8:0: C0305: Trailing newlines (trailing-newlines)
my_script.py:1:0: C0114: Missing module docstring (missing-module-docstring)
my_script.py:3:0: C0103: Constant name "a" doesn't conform to UPPER_CASE naming style (invalid-name)

---

Your code has been rated at 0.00/10 (previous run: 0.00/10, +0.00)

pylint --disable=C0114 my_script.py     deshabilitar ciertos verificadores

#----------Comentar codigo----------
# Asignando 1 a alguna variable; hace perder el tiempo
a = 1

# Inicializando el contador de ciclos en 1; más útil
a = 1

#----------docstrings ----------
the first statement in a module, function, class, or method definition
always use """triple double quotes""" around docstrings. 
It prescribes the function or method’s effect as a command (“Do this”, “Return that”), not as a description
def function(a, b):
    """Do X and return a list."""

Es un texto que explica qué hace, qué necesita y qué devuelve tu función
def calc(x, y):     #confuso
    return x * y + (x * 0.16)

def calcular_precio_con_iva(precio, cantidad):
    """
    Calcula el precio total incluyendo IVA del 16%.
    
    Args:
        precio (float): Precio unitario del producto
        cantidad (int): Cantidad de productos
        
    Returns:
        float: Precio total con IVA incluido
        
    Example:
        >>> calcular_precio_con_iva(100, 2)
        232.0
    """
    subtotal = precio * cantidad
    iva = subtotal * 0.16
    return subtotal + iva

#----------pytest----------
pytest sign.py      ejecutar
pytest -rA age_group.py     -rA significa que pedimos a pytest que muestre todos los resúmenes de las pruebas

ejecutar pruebas explícitamente
filtrar qué casos de prueba ejecutar
volver a ejecutar solo las pruebas fallidas
informar los resultados de las pruebas
Las pruebas unitarias se utilizan para probar partes individuales de nuestro código

busca en tu código funciones con nombres que comiencen con test_, y ejecuta esas funciones 
Si no se genera una AssertionException dentro de una función de prueba, se considera aprobada

# Crea la función sign()
def sign(x):
    """Devuelve el signo de número."""
    if x == None:
        return None
    if x < 0:
        return -1
    return 1

# Prueba la función sign()
def test_sign():
    assert sign(-10) == -1
    assert sign(10) == 1
    assert sign(0) == 1
    assert sign(None) == None

assert expression, "mensaje de aserción"
debe producir True si no hay ningún problema, de lo contrario, False
la excepción AssertionError se lanza con un método de aserción si se proporciona uno. (False)

#----------Clonar repositorios----------
Ctrl+Shift+P
Git: Clone
https://github.com/Carolina-Lezama/experimento_uno.git   link de el repositorio










































































































































































































import pandas as pd
import numpy as np
import plotly.express as px

cases = [33, 61, 86, 112, 116, 129, 192, 174, 344, 304, 327, 246, 320, 339, 376]

dates = ['March<br>'] * len(cases)
day = 18
for i in range(len(dates)):
    dates[i] = dates[i] + str(day)
    day = day + 1
dates[-1] = 'April<br>1'

labels = dict(date="Date", cases="Number of cases")
markers = dict(size=30, line=dict(width=2, color='black'), color='white')
title = dict(text='New Cases Per Day', font=dict(color='white', size=30))
yaxis = dict(tickmode='linear', tick0=30, dtick=30)

df = pd.DataFrame({'cases': cases, 'date': dates})

fig = px.line(df, y='cases', x='date', text='cases', markers=True, labels=labels, title="New Cases Per Day")

fig.update_xaxes(showgrid=False, color='white', tickangle=0)
fig.update_yaxes(color='white', gridcolor='#5c5a5c', gridwidth=2, range=[15, 400])
fig.update_traces(marker=markers, line_color='white', line_width=6)
fig.update_layout(title=title,
                  title_x=0.5,
                  paper_bgcolor='#070230',
                  plot_bgcolor='#070230',
                  yaxis=yaxis,
                  xaxis_type='category')
fig.add_annotation(text='TOTAL CASES', 
                    align='right',
                    showarrow=False,
                    font=dict(color='white', size=12),
                    xref='paper',
                    yref='paper',
                    x=1.08,
                    y=1.25)
fig.add_annotation(text='3,342', 
                    align='right',
                    showarrow=False,
                    font=dict(color='white', size=23),
                    xref='paper',
                    yref='paper',
                    x=1.071,
                    y=1.2)

fig.show()