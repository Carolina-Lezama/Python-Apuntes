#------------DATO DE TIPO STRING-------------------
a = "Hola, Python"
b = """Usar varias comillas permite 
        hacer saltos de linea
        como ahora 
    """

#------------DATO DE TIPO ENTERO-------------------
c = 10

#------------DATO DE TIPO FLOTANTE-----------------
d = 10.5


#------------DATO DE TIPO BOOLEANO-----------------
e = True
f = False

#------------FORMAS DE SUMAR-RESTAR-----------------
num = 0

num = 5 + 5
num += 5

num = 5 - 5
num -= 5
#+=, -= y *= /= 

#------------CONCATENACION-----------------
stri1 = "cadenas"
stri2 = 'Unir ' + stri1 + " es genial"

stri2 = f"Sea cualquier tipo {stri1} F es capaz de volver un dato a texto" 

#------------BORRAR VARIABLES-----------------
del a

#------------BUSCAR CADENA (DEVUELVE BOOLEAN)-----------------
ejemplo = 'Esta cadena contiene informacion'

resultado = 'contiene' in ejemplo
resultado = 'contiene ' not in ejemplo

#------------LISTAS-----------------
Lista = [1, 2, 3, 4, 5]
print(Lista) 
print(Lista[0])  # Acceso al primer elemento

#------------TUPLAS (NO SE PUEDEN MODIFICAR)-----------------
Tupla = (1, 2, 3, 4, 5)
Tupla_dos = "Dato",1,True
Tupla_dos = "1 solo dato",
print(Tupla) 
print(Tupla[0])

#------------CONJUNTOS SET (NO SE PUEDEN MODIFICAR POR ELEMENTO PERO SI REESCRIBIRLA)-----------------
Conjunto = {1, 2, 3, 4, 5, 5}  #No permite duplicados
Conjunto = {5} #reescribiendo
#Se recorre con bucle

#------------EXPONENTE-----------------
exponente = 2 ** 3 

#------------DIVISION SIN RESIDUO-----------------
division = 7 // 3 #la divion normal devuelve float; este devuelve int

#------------MODULO-----------------
modulo = 7 % 3 

#------------OPERADORES DE COMPARACION (Devuelve booleano) -----------------
a = 5 == 5
b = 5 != 3
c = 5 > 3
d = 5 < 3
e = 5 >= 3
f = 5 <= 3

#------------CONDICIONALES-----------------
#se usan los booleanos para revisar si se cumple o no la condicion
condicional = 16
if condicional >= 18:
    print("Cumplio condicion")
elif condicional == 17:
    print("Cumplio condicion alternativa")
else:
    print("No cumplio condicion")

if "cadena " in ejemplo:
    print("La cadena esta en el ejemplo")

#------------OPERADORES LOGICOS-----------------
resultado = True and True #Ambos deben ser true
resultado = True & True

resultado = True or False #Al menos uno deben ser true
resultado = True | False

resultado = not True #Invierte valor
resultado = not False

#------------METODOS DE CADENAS-----------------
oracion = 'Aqui buscaremos variable en el string'
cadena = ' variable '

cadena.upper() # .lower ; .capitalize
oracion.find(cadena) # Devuelve posicion de la cadena; sino encuentra devuelve -1
cadena.isnumeric() #da booleano
oracion.count(cadena) #Cuenta cuantas veces aparece la cadena
len(oracion) #Contar caracteres, los espacios cuentan
cadena.replace('vari', 'cari')
oracion.startswith(cadena) #.endswith, verificar si un texto empieza/termina con cierta cadena
oracion.split(' ') #Dividir cada que encuentre caracter (espacio), devuelve lista 

words = ['Hola', 'mundo']
phrase = ' '.join(words)  # Une lista de palabras en una cadena separada por espacios

cadena = cadena.strip()  # Elimina espacios al inicio y al final

#------------METODOS DE LISTAS-----------------
Lista = [1, 2, 3, 4, 5]
len(Lista) #Contar elementos
Lista.append("Agregar cualquier tipo de dato, al final")
Lista.insert(2,True) #Indice, Valor
Lista.extend([True, 5, "Michi"]) #Unir una lista con otra

#posicion 0 igual a primer dato de la lista
guardar_elemento_borrado = Lista.pop(5) #Eliminar por indice; -1 es el ultimo
Lista.remove("Cualquier tipo de dato, al final") #por valor
Lista.clear() #Eliminar todos los elementos
Lista.sort(reverse=True) #Ordenar, metodo, puntuación, números, A-Z, a-z
lista_nueva = sorted(Lista, reverse=True) #Ordenar y crear una nueva lista
Lista.reverse() #Invertir el orden de los valores

Lista.index(True) #Devuelve indice del valor buscado
Lista.index(0) #solo devuelve la posición de la primera coincidencia, Si el valor no existe, lanza un error
lista = [1, 2, 3, True]
print(lista.index(True))  # True se considera igual a 1 (y False igual a 0)

a = [1, 2, 3] #Concatenar listas
b = [4, 5, 6]
c = a + b
a = [1, 2, 3] #Repetir una lista
print(a*2) #solo impresion, no se guarda la variable
Lista = [1, 2, 3, 4, 5, [6, 7, 8]]
a = Lista[4:8] #si los valores no existen, no da error, solo toma hasta el final

#Daria error si porque no se puede comparar list ni str con int
min(Lista) #valor minimo
max(Lista) #valor maximo
sum(Lista) #suma todos los valores numericos

#------------DICCIONARIO clave-valor ----------------
Diccionario = {
    0 : "Hola",
    'variable' : "Python",
}
print(Diccionario['variable']) 

#------------METODOS DE DICCIONARIOS-----------------
Diccionario.keys() #Devuelve las claves; .values() devuelve los valores
Diccionario.get(0,"valor devuelto si no se encuentra") #devuelve valor de una key
    # Busca en el diccionario la clave 0.
    # Si existe → devuelve su valor verdadero.
    # Si NO existe → devuelve "valor devuelto si no se encuentra".

Diccionario.clear() #Eliminar todos los pares clave-valor
guardar_valor_borrado = Diccionario.pop('variable', "valor dado por si no existe") #Solo borra 1
valor = Diccionario["variable"]
Diccionario.items()	#Devuelve una lista que contiene una tupla para cada par clave-valor
Diccionario.append("Nueva clave", "nuevo valor") #Agregar nuevo par clave-valor
Diccionario.update({'Nueva Clave': 'Nuevo Valor'}) #Actualizar o agregar par clave-valor
del Diccionario['variable'] #Eliminar par clave-valor, se pone la clave a eliminar
Diccionario["Nuevo"] = "Valor" #Otra forma de agregar nuevo par clave-valor
    
#------------INPUTS-----------------
guardable = int(input("Ingresar un dato: "))

#------------DESEMPAQUETADO-----------------
datos_lista = [1, 2, 3]
datos_tupla = (1, 2, 3)

a, b, c = datos_lista
x, y, z = datos_tupla

#------------TEORIA DE CONJUNTOS SET-----------------
conjunto_uno = {1,2,3,4}
conjunto_dos = {2,4}

resultado = conjunto_dos.issubset(conjunto_uno) #Verifica si es subconjunto
resultado = conjunto_dos <= conjunto_uno

resultado = conjunto_uno.issuperset(conjunto_dos) #Verifica si es superconjunto
resultado = conjunto_uno > conjunto_dos 

#------------BUCLE FOR----------------
animales = ["Perro", "Gato", "Pájaro"]
numeros = [1, 2, 3]
for animal,numero in zip(animales, numeros):
    #zip() une dos (o más) listas elemento por elemento, formando pares.
    #luego se desempaqueta en el for
    print(animal)
    print(numero)

for num in range(5, 10): #Del 5 al 9
    print(num) 

for indice, valor in enumerate(numeros): #Devuelve tupla con indice y valor
    print(num) 
    print(num[0]) #Devuelve solo indices
    print(num[1]) #Devuelve solo valores

for animal in animales:
    if animal == "Gato":
        continue  # Salta al siguiente ciclo si es "Gato"
    if animal == "Perro":
        break  # Sale del bucle si es "Perro"
    print(animal)
else:
    print("Siempre se imprimira a menos que se use break")

cadena = "Hola Mundo"
for letra in cadena:
    print(letra)  # Imprime cada letra de la cadena

#------------list comprehension----------------
a = [x*2 for x in numeros]  
pares = [x for x in numeros if x % 2 == 0]
dobles_pares = [x*2 for x in numeros if x % 2 == 0]

#------------BUCLE WHILE----------------
contador = 0
while contador < 5:
    print(contador)
    contador += 1  # Incrementa el contador en 1

#------------FUNCIONES INTEGRADAS----------------
round(23.53433, 4) #Cantidad de decimales

#------------CREAR Y LLAMAR FUNCIONES ----------------
def saludar(nombre): #pide argumentos
    print(f"Hola, {nombre}!")
saludar("Juan")  # Llama a la función con argumento

def suma(nombre, *edad): #* para varios argumentos, se guardan en tupla
    return f"{nombre} tiene {sum(edad)} años en total."
resultado = suma("Ana", 5, 10, 15 )

def frase(parametro_1, parametro_2, parametro_3 = "Parametro forzado" ): #parametro por defecto
    return f"Hola {parametro_1} {parametro_2} y {parametro_3} !"
uso = frase(parametro_2 = "Juan", parametro_3 = "Pedro", parametro_1 = "Maria")

#------------FUNCION LAMBDA ----------------
resultado = lambda x: x + 10
print(resultado(5))  # Imprime 15

numeros_pares = list(filter(lambda x: x % 2 == 0, Lista))  # Filtra números pares
    # Recorre cada elemento del iterable (tu lista).
    # Aplica la función.
    # Se queda solo con los elementos donde la función devuelve True.
print(numeros_pares) 

#------------MODULOS----------------
import Scripts.age_group #importar modulo de otra carpeta
    # Scripts/
    #     age_group.py
Uso = Scripts.age_group.test_get_age_group("Parametro, no existe funcion")

import Sprint_3 as S3 #importar modulo y renombrar, dentro de la misma carpeta
Uso = S3.replace_wrong_values("Parametro, no existe funcion")

from Sprint_4 import double_it #importar funcion especifica de modulo
Uso = double_it("Parametro, no existe funcion")

#------------SUBCANDENAS----------------
cadena = "Hola, mundo!"
subcadena = cadena[0:4]  # Extrae "Hola" no  forma parte la , aunque este en el indice
subcadena = cadena[0:] #Tomar todo
subcadena = cadena[:5]  # Extrae "Hola," hasta el índice 4
subcadena = cadena[-6:1000]  # No causara error, solo se tomara hasta el final
    #resultado: " mundo!"
subcadena = cadena[-6:-8]  # De forma negativa
    #resultado: "" (cadena vacía, ya que el inicio es mayor que el fin)

#------------FORMAT----------------
mensaje = "Hola, {}. Tienes {} mensajes.".format("Juan", 5)
mensaje = "Hola, {nombre}. Tienes {mensajes} mensajes.".format(nombre="Juan", mensajes=5)

#------------USO DE CARACTERES DE ESCAPE----------------
cadena = "Hola, \"mundo\"!"  # Usa comillas dobles dentro de una cadena con comillas dobles
#\t para simular el espacio de un tab
#\n para salto de linea 
#\ para caracteres especiales

#------------FUNCIONES----------------
def Funcion(parametro):
    return f"Hola, {parametro}!"