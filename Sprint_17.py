#---------------------------- Machine learning para textos ----------------------------
# Descubrir cómo convertir textos en números para que se puedan usar con algoritmos de machine learning.

# Las personas usan texto para comunicarse
# Las empresas usan el análisis de texto para evaluar con el fin de saber qué piensan sus clientes acerca de

# 1. convertir el texto que deseas analizar a un formato que sea compatible con el machine learning

#---------------------------- Procesamiento de texto ----------------------------
# Antes de extraer características, vamos a tener que simplificar el texto
# Ayudará a que el texto sea más fácil de manejar

# Pasos para el preprocesamiento de texto:
#     1. Tokenización: dividir el texto en tokens (frases, palabras y símbolos separados)
#     2. Lematización: reducir las palabras a sus formas fundamentales (lema)

#Importar librerías
from nltk.tokenize import word_tokenize
# ¿Qué hace? Tokenización = dividir texto en palabras/tokens individuales

# Ejemplo:
texto = "Hello world! How are you?"
tokens = word_tokenize(texto)
print(tokens)
# Output: ['Hello', 'world', '!', 'How', 'are', 'you', '?']

# Características:
# - Separa palabras por espacios
# - Maneja puntuación como tokens separados
# - Reconoce contracciones (don't → ['do', "n't"])

from nltk.stem import WordNetLemmatizer
# ¿Qué hace? Lematización = reducir palabras a su forma base/raíz (lemma)

# Ejemplo:
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running"))  # → "running" (necesita POS tag)
print(lemmatizer.lemmatize("running", pos='v'))  # → "run" (verbo)
print(lemmatizer.lemmatize("better", pos='a'))   # → "good" (adjetivo)

# Diferencia clave:
# Tokenización: "Hello world!" → ['Hello', 'world', '!']
# Lematización: ['running', 'cats', 'better'] → ['run', 'cat', 'good']

# ¿Por qué se usan juntas?
# 1. Primero tokenizas: texto → lista de palabras
# 2. Luego lematizas: cada palabra → forma base

#---------------------------- ¿Qué es un POS tag (Part-of-Speech tag)? ----------------------------
# Es una etiqueta que identifica la categoría gramatical de una palabra en un texto.
# Dice si una palabra es un sustantivo, verbo, adjetivo, adverbio, etc.

# "run" puede ser un verbo ("I run") o un sustantivo ("a morning run")
# "fast" puede ser un adjetivo ("fast car") o un adverbio ("run fast")

# ¿Por qué son importantes en lemmatización?
# La lemmatización necesita conocer el POS tag porque la misma palabra puede tener diferentes lemas según su función gramatical

# Cuando usas spaCy, cada token automáticamente recibe un POS tag. Puedes acceder a él con token.pos_

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("The cats are running fast")
for token in doc:
    print(f"{token.text} → POS: {token.pos_} → Lema: {token.lemma_}")


#---------------------------- Lematización ----------------------------
#pasar el texto como tokens separados 
lemmatizer  = WordNetLemmatizer()
text = "All models are wrong, but some are useful."
tokens = word_tokenize(text.lower()) 
    #word_tokenize siempre lleva el texto como parametro
    # word_tokenize() es una función, no un objeto. No puedes llamarla sin parámetros
lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # devuelve el lema de un token que se le pasó
    # el resultado generalmente se presenta como una lista de tokens lematizados.
    # convertimos el texto a minúsculas porque así lo exige el lematizador NLTK.

#output: ['all', 'model', 'are', 'wrong', ',', 'but', 'some', 'are', 'useful', '.']

" ".join(lemmas) # volver a convertir la lista de tokens procesados en una línea de texto
#output: 'all model be wrong , but some be useful .'

#---------------------------- ¿Qué es una pipeline? (tubería) ----------------------------
# Una pipeline es una secuencia de pasos o procesos que se ejecutan uno tras otro, donde la salida de un paso se convierte en la entrada del siguiente.

# Analogía del mundo real
# Paso 1: Se ensambla el chasis
# Paso 2: Se instala el motor (usando el chasis del paso 1)
# Paso 3: Se pintan las piezas (usando el auto parcial del paso 2)
# Paso 4: Se instalan las ruedas (usando el auto pintado del paso 3)

# Cada paso depende del anterior y alimenta al siguiente.

# En programación y datos
# datos_originales → limpieza → transformación → análisis → resultados

#---------------------------- ¿Qué es un lemma? ----------------------------
# Un lemma es la forma base o raíz de una palabra. 
# Es como el "nombre original" de la palabra en el diccionario.

# Ejemplos:
# - "running", "ran", "runs" → lemma: "run"
# - "better", "best" → lemma: "good"
# - "am", "is", "are", "was", "were" → lemma: "be"

# Cuando analizas texto, quieres que "corriendo" y "corrió" se traten como la misma idea: correr.

#---------------------------- ¿Qué es una list comprehension? ----------------------------
# forma compacta y elegante de crear listas en Python

# Forma tradicional (con bucle):
lemmas = []
for token in doc:
    lemmas.append(token.lemma_)

# Con list comprehension:
lemmas = [token.lemma_ for token in doc]

#---------------------------- spaCy ----------------------------
# nlp significa "procesamiento del lenguaje natural" y contiene un modelo principal para el reconocimiento de texto.
# spaCy utiliza una pipeline especial para procesar el texto entrante

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) 
    #indica cargar el modelo de tamaño pequeño en inglés y deshabilitar algunos componentes 
doc = nlp(text.lower())
    # nlp Procesa el texto con el modelo de spaCy, Doc que contiene todos los tokens analizados, Cada token tiene información lingüística 
lemmas = [token.lemma_ for token in doc]
    # Recorre cada token en el documento procesado
    # Extrae el lema de cada token usando .lemma_
    # Crea una lista con todos los lemas
print(" ".join(lemmas))
#output: 'all model be wrong , but some be useful .'

# Ejemplo práctico - texto: "I am running quickly"

# Después de doc = nlp(text.lower())
# doc contiene tokens: ["i", "am", "running", "quickly"]

# Después de lemmas = [token.lemma_ for token in doc]
# lemmas = ["i", "be", "run", "quickly"]

#normalmente los textos tienen un "código de tonalidad" que implica una tono positivo o negativo

#---------------------------- Importa los datos ----------------------------
import pandas as pd
data = pd.read_csv('/datasets/imdb_reviews_small.tsv', sep='\t')

# TSV significa "valores separados por tabuladores".
# TSV usa el carácter de tabulación como delimitador en lugar del carácter de coma.

# es necesario definir un corpus para una tarea de análisis de texto
# Cada registro de texto es tratado por un algoritmo de machine learning, de acuerdo con su "posición" en un corpus. 

#---------------------------- ¿Qué es un corpus? ----------------------------
# es una colección grande y estructurada de textos que se utiliza para análisis y procesamiento de lenguaje natural.
# una "biblioteca digital" de documentos que los investigadores y programadores usan para entrenar modelos y realizar análisis de texto

# Un conjunto de textos se denomina colectivamente corpus

corpus = data['review']

#---------------------------- Ejemplo aplicado ----------------------------
import pandas as pd
import random  # para seleccionar una reseña aleatoria
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data = pd.read_csv('/datasets/imdb_reviews_small.tsv', sep='\t')
corpus = data['review']
def lemmatize(text):
    doc = nlp(text.lower())
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)
review_idx = random.randint(0, len(corpus) - 1)
review = corpus[review_idx]
print('El texto original:', review)
print()
print('El texto lematizado:', lemmatize(review))

#---------------------------- ¿Qué es spaCy? ----------------------------
# spaCy es una librería de procesamiento de lenguaje natural (NLP) muy potente y rápida.

# ¿Para qué sirve?
# Tokenización
# Lematización
# Análisis gramatical (POS tagging) 
#     Identifica la función gramatical de cada palabra (sustantivo, verbo, adjetivo, etc.)
# Reconocimiento de entidades (NER)
#     Identifica y clasifica entidades importantes como nombres de personas, lugares, organizaciones, fechas, etc.
# Análisis sintáctico (parsing)
#     Analiza la estructura de la oración y las relaciones entre palabras (quién hace qué a quién).
# ...

#---------------------------- Importación y configuración ----------------------------
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Carga el modelo de inglés pequeño
# Desactiva componentes para mayor velocidad
#     parser = análisis sintáctico
#     ner = reconocimiento de entidades

#---------------------------- ¿Qué es un objeto doc? ----------------------------
# Un documento procesado que contiene toda la información del texto analizado.

# Crear el objeto doc
texto = "Hello world! I am learning spaCy."
doc = nlp(texto)

# Iterar sobre tokens
for token in doc:
    print(f"Palabra: {token.text}")
    print(f"Lema: {token.lemma_}")
    print(f"POS: {token.pos_}")

# Output:
# Palabra: Hello
# Lema: hello
# POS: INTJ
# Palabra: world
# Lema: world  
# POS: NOUN

#---------------------------- Ventaja de spaCy vs NLTK ----------------------------
# NLTK: Necesitas múltiples pasos
# spaCy: Todo en un mismo paso

#---------------------------- Pipeline completa de spaCy, ejecuta estos componentes en orden ----------------------------
# Tokenizer: Divide el texto en palabras
#     se debe caseonvertir el texto a minúsculas
# Tagger: Asigna POS tags (sustantivo, verbo, etc.)
# Parser: Analiza la estructura sintáctica
#     Extraer los lemas de cada token
# NER: Reconoce entidades (nombres, lugares, etc.)
# Lemmatizer: Encuentra la forma base de cada palabra

#---------------------------- expresiónes regulares ----------------------------
# encontrar patrones complejos en los textos.
# Las expresiones regulares encuentran secuencias de caracteres, palabras y números mediante el reconocimiento de patrones.

# módulo  para trabajar con expresiones regulares
import re

# patrón: la secuencia de caracteres que deseas encontrar
# sustitución: con qué debe sustituirse cada coincidencia de patrón
# texto: el texto que la función escanea en busca de coincidencias de patrón
re.sub(pattern, substitution, text)
    # encuentra todas las partes del texto que coinciden con el patrón dado y luego las sustituye con el texto elegido.

# Las expresiones regulares tienen su propia sintaxis     
# suelen utilizar el carácter ('\') como parte de su sintaxis
# las expresiones regulares se definen usando cadenas sin formato

# cadena normal
print('¡Hola!\n')
# ¡Hola!

# cadena sin formato (raw string)
print(r'¡Hola!\n')
# ¡Hola!\n
# definimos una cadena sin formato escribiendo r antes de la cadena

# Ejemplo: "a.b" (coincide con "a", seguido de cualquier carácter, seguido de "b")(cadena de tres caracteres)
# . indica que cualquier carácter puede aparecer en esa posicion, excepto un salto de línea.

# Como parte del paso de preprocesamiento, debemos eliminar todos los caracteres excepto las letras, los apóstrofos y los espacios

# El patron a bus car se enlistan entre corchetes, sin espacios, y se pueden colocar en cualquier orden
# un rango de letras se indica con un guión:

text = """
I liked this show from the first episode I saw, which was the "Rhapsody in Blue" episode (for those that don't know what that is, the Zan going insane and becoming pau lvl 10 ep). Best visuals and special effects I've seen on a television series, nothing like it anywhere.
"""

pattern = r"[a-zA-Z]" # coincide con cualquier letra mayúscula o minúscula
pattern = r"[a-zA-Z']" # incluye apóstrofos
pattern = r"[^a-zA-Z']" # cualquier caracter que no sea una letra o un apóstrofo (el símbolo ^ dentro de los corchetes indica negación)
# (^ al comienzo de la secuencia)

text = re.sub(pattern, " ", text)
print(text)

#---------------------------- eliminar los espacios adicionales ----------------------------
text = text.split()
print(text)
#resultado: una lista sin espacios - ['I', 'liked', 'this', 'show', 'from', 'the', 'first', 'episode', 'I', 'saw', 'which', 'was', 'the', 'Rhapsody', 'in', 'Blue', 'episode', 'for', 'those', 'that', "don't", 'know', 'what', 'that', 'is', 'the', 'Zan', 'going', 'insane', 'and', 'becoming', 'pau', 'lvl', 'ep', 'Best', 'visuals', 'and', 'special', 'effects', "I've", 'seen', 'on', 'a', 'television', 'series', 'nothing', 'like', 'it', 'anywhere']

#---------------------------- Recombinar en una cadena  con espacios ----------------------------
text = " ".join(text)
print(text)
# resultado: "I liked this show from the first episode I saw which was the Rhapsody in Blue episode for those that don't know what that is the Zan going insane and becoming pau lvl ep Best visuals and special effects I've seen on a television series nothing like it anywhere" 

#---------------------------- Ejemplo completo ----------------------------
import random  # para seleccionar una reseña aleatoria
import pandas as pd
import re
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data = pd.read_csv('/datasets/imdb_reviews_small.tsv', sep='\t')
corpus = data['review']
def clear_text(text):
    pattern = r"[^a-zA-Z']"
    text = re.sub(pattern, " ", text)
    text = text.split()
    text = " ".join(text)
    return text
def lemmatize(text):
    doc = nlp(text.lower())
    lemmas = []
    for token in doc:
        lemmas.append(token.lemma_)
    return ' '.join(lemmas)
review_idx = random.randint(0, len(corpus)-1)
review = corpus[review_idx]
print('El texto original:', review)
print()
print('El texto lematizado:', lemmatize(clear_text(review)))

#---------------------------- Bolsa de palabras y n-gramas ----------------------------
# cómo podemos convertir datos de texto en datos numéricos
# La conversión generalmente se realiza para que una palabra o una oración se convierta en un vector numérico

# "bolsa de palabras": transformar textos en vectores sin tomar en cuenta el orden de las palabras

#Original:
    # For want of a nail the shoe was lost.
    # For want of a shoe the horse was lost.
    # For want of a horse the rider was lost.

#No mayusculas y lemantizado
    # for want of a nail the shoe be lose 
    # for want of a shoe the horse be lose 
    # for want of a horse the rider be lose

#Contar cuántas veces aparece cada palabra
    # "for," "want," "of," "a," "the," "be," "lose" — 3;
    # "shoe," "horse" — 2;
    # "nail," "rider" — 1.

#Diccionario de palabras y conteos, colacadas en orden alfabetico
[3, 3, 3, 2, 3, 1, 3, 1, 2, 3, 3]

# Otra forma de hacerlo:
import spacy
from collections import Counter
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
text = """For want of a nail the shoe was lost. For want of a shoe the horse was lost. For want of a horse the rider was lost."""
doc = nlp(text)
    # 1. Tokenización: dividir el texto en palabras/tokens individuales
    # 2. Lematización: reducir las palabras a sus formas fundamentales (lema)
    # 3. Creación del objeto Doc: El resultado es un objeto Doc que contiene todos los tokens analizados.
tokens = [token.lemma_ for token in doc if not token.is_punct]
    # 1. token.lemma_ - Extrae el lemma (forma base) de cada token
    #     - Ejemplo: "running" → "run
    # 2. for token in doc - Recorre cada token en el documento procesado
    # 3. if not token.is_punct - Filtra los tokens que NO son signos de puntuación
    #     - token.is_punct es True para: ".", ",", "!", "?", etc.
    #     - not token.is_punct significa: "solo si NO es puntuación"
bow = Counter(tokens) #crea una bolsa de palabras
    # Crea un diccionario donde las claves son las palabras y los valores son las frecuencias
    # Ejemplo: ['for', 'want', 'of', 'for', 'want']
    # Resultado: Counter({'for': 2, 'want': 2, 'of': 1})
vector = [bow[token] for token in sorted(bow)] #crea un vector ordenado alfabéticamente
print(vector)
#Resultado: [3, 3, 3, 2, 3, 1, 3, 1, 2, 3, 3]

#---------------------------- Varias frases analizadas ----------------------------
# Si hay varios textos, la bolsa de palabras los transforma en una matriz. 
# Los números en sus intersecciones representan ocurrencias de cada palabra única.
# Las filas representan textos y las columnas representan palabras únicas de todos los textos del corpus.

#ejemplo:
# Texto 1: for want of a nail the shoe be lose 
# Texto 2: for want of a shoe the horse be lose
# Texto 3: for want of a horse the rider be lose

# Resultado:
#           for    want	   of	 a	  nail	   the  	shoe	be	lose	horse	rider
# Texto 1	 1	    1	    1	 1      1	    1	      1	     1	  1	      0	      0
# Texto 2	 1	    1	    1	 1   	0	    1	      1	     1	  1	      1	      0
# Texto 3	 1	    1	    1	 1   	0	    1	      0	     1	  1	      1	      1

# La bolsa de palabras puede contener palabras en cualquier orden
# el orden debe ser el mismo para todos los textos (filas de matriz) para que se puedan comparar los vectores.
# La bolsa de palabras cuenta cada palabra única, pero no considera el orden de las palabras ni las conexiones entre ellas

#---------------------------- n-gramas ----------------------------
# es una secuencia de varias palabras. 
# N indica el número de elementos y es arbitrario.
# N=1, tenemos palabras separadas o unigramas. 
# N=2, tenemos frases de dos palabras o bigramas
# N=3 produce trigramas. 

#---------------------------- ¿Qué es un n-grama? ----------------------------
"Me gusta mucho el chocolate"

# tomar grupos de palabras consecutivas de esa oración. La "N" es solo un número que nos dice cuántas palabras tomamos juntas cada vez.

# Los diferentes tipos:
# N=1 (Unigramas): Tomamos las palabras una por una
"Me"
"gusta"
"mucho"
"el"
"chocolate"

# N=2 (Bigramas): Tomamos las palabras de dos en dos
"Me gusta"
"gusta mucho"
"mucho el"
"el chocolate"

# N=3 (Trigramas): Tomamos las palabras de tres en tres
"Me gusta mucho"
"gusta mucho el"
"mucho el chocolate"

# ¿Por qué es útil esto?
# si solo tenemos palabras sueltas como "Peter", "viaja", "de", "Tucson", "a", "Vegas", no sabemos el orden ni la relación entre ellas.

#---------------------------- Creacion de una bolsa de palabras ----------------------------
#convertir un corpus de texto en una bolsa de palabras:

# Importa la clase
from sklearn.feature_extraction.text import CountVectorizer

# Crea un contador
count_vect = CountVectorizer()

# bow = bolsa de palabras
bow = count_vect.fit_transform(corpus)
# Los números en sus intersecciones representan cuántas veces aparece una determinada palabra en el texto
# extrae palabras únicas del corpus y cuenta cuántas veces aparecen en cada texto del corpus
# devuelve una matriz donde las filas representan textos y las columnas muestran palabras únicas del corpus

# descubrir el tamaño de la matriz
bow.shape

#bolsa de palabras en forma de matriz
print(bow.toarray())

# La lista de palabras únicas en la bolsa se llama vocabulario

#ver el nombre de las palabras
count_vect.get_feature_names()

# calculos de n-gramas, especificar el tamaño del n-grama con el argumento: ngram_range
# establecer el tamaño mínimo y máximo de n-gramas
count_vect = CountVectorizer(ngram_range=(2, 2)) #bigramas 

# un corpus grande representa una bolsa de palabras más grande
# por lo general, puedes eliminar las conjunciones y las preposiciones sin perder el significado de la oración

#---------------------------- encontrar palabras vacías (palabras que no significan nada por sí solas). ----------------------------
# asegurar de obtener una bolsa de palabras más limpia

import nltk #descargar el paquete
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) #obtener un conjunto de palabras vacías para inglés
count_vect = CountVectorizer(stop_words=stop_words) #pasar lista de palabras vacias
#el contador sabe qué palabras se deben excluir de la bolsa de palabras

#---------------------------- Ejemplo de bolsa de palabras con y sin vacias ----------------------------
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
corpus = data['review_lemm']
stop_words = set(stopwords.words('english'))
count_vect_stop_words = CountVectorizer(stop_words=stop_words) 
count_vect = CountVectorizer() 
bow_1 = count_vect_stop_words.fit_transform(corpus)
bow_2 = count_vect.fit_transform(corpus)
print('El tamaño de BoW con palabras vacías:', bow_2.shape)
print('El tamaño de BoW sin palabras vacías:', bow_1.shape)

#resultado:
# El tamaño de BoW con palabras vacías: (4541, 26214)
# El tamaño de BoW sin palabras vacías: (4541, 26098)

#---------------------------- Ejemplo de contador de n-gramas para el corpus  ----------------------------
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
corpus = data['review_lemm']
count_vect = CountVectorizer(ngram_range=(2, 2)) 
n_gram = count_vect.fit_transform(corpus)
print('El tamaño del bigrama:', n_gram.shape)

#resultado:
# El tamaño del bigrama: (4541, 322321)

#---------------------------- TF-IDF (frecuencia de término — frecuencia inversa de documento) ----------------------------
# La bolsa de palabras toma en cuenta la frecuencia de las palabras.
# puede crear un problema cuando: una palabra en particular no se usa con frecuencia, pero contiene mucha información
# la bolsa de palabras puede fallar al priorizar una palabra importante

# La importancia de una palabra se determina por el valor de TF-IDF 
# TF es la frecuencia con la que aparece una palabra en un texto
# IDF mide su frecuencia de aparición en el corpus.

# Esta es la fórmula para TF-IDF:
# TF−IDF=TF×IDF

# Así es como se calcula TF:
# TF= t/n
# t = número de ocurrencias de palabras 
# n = número total de palabras en el texto

# Calcular IDF
# IDF = log10D/d
# D = total de documentos en el corpus
# d = documentos que contienen la palabra

# El papel de la IDF en la fórmula es reducir el peso de las palabras más utilizadas en cualquier otro texto del corpus dado
# La IDF depende del número total de textos en un corpus (D) y el número de textos donde aparece la palabra (d).

# Cuanto mayor sea TF-IDF, más única será la palabra
# Si una palabra aparece en MÁS documentos del corpus,  TF-IDF disminuye

# Aunque "almeja(o cualquier otra palabra)" aparezca varias veces en el texto, no es relevante porque:
# Aparece en todos los textos del corpus
# No ayuda a distinguir un texto de otro
# No aporta información única
# Es como si fuera una "palabra vacía" en este contexto específico.

# recordemos que el calor mientras mas alto es, mas especial es la palabra, y mientras mas bajo es, mas común es la palabra.

#---------------------------- TF-IDF en sklearn ----------------------------
#importar libreria
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('spanish'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words) #contador y palabras vacias
tf_idf = count_tf_idf.fit_transform(corpus) #alcular la TF-IDF para el corpus de texto

# Podemos calcular los n-gramas al pasar el argumento ngram_rangeso de entrenamiento. 
# Si los datos se dividen en conjuntos de entrenamiento y prueba, llama a la función fit() solo para el conjunto de entrenamiento

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as nltk_stopwords
data = pd.read_csv('/datasets/imdb_reviews_small_lemm.tsv', sep='\t')
corpus = data['review_lemm']
stop_words = set(nltk_stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words) #contador y palabras vacias
tf_idf = count_tf_idf.fit_transform(corpus)
print('El tamaño de la matriz TF-IDF:', tf_idf.shape)
#Resultado: El tamaño de la matriz TF-IDF: (4541, 26098)

#---------------------------- Analisis de sentimiento ----------------------------
# identifica textos cargados de emociones
# El análisis de sentimiento es una tarea de clasificación donde queremos predecir si un texto es positivo, negativo o neutral. 
# Para determinar el tono del texto, vamos a usar valores TF-IDF como características.
# útil al evaluar las reacciones de los consumidores ante un producto.

# El análisis de sentimiento funciona etiquetando el texto como positivo o negativo.
# Al texto positivo se le asigna un "1" y al texto negativo se le asigna un "0".

# El análisis de sentimiento NO está limitado solo a regresión, es decir tambien aplica a los modelos de ensable

#ejemplo:
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv('/datasets/imdb_reviews_small_lemm_train.tsv', sep='\t')
test_data = pd.read_csv('/datasets/imdb_reviews_small_lemm_test.tsv', sep='\t')
train_corpus = train_data['review_lemm'] 
stop_words = set(nltk_stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stop_words)
tf_idf = count_tf_idf.fit_transform(train_corpus)
features_train = tf_idf
target_train = train_data['pos']
test_corpus = test_data['review_lemm'] 
tf_idf_test = count_tf_idf.transform(test_corpus)
features_test = tf_idf_test
model = LogisticRegression()
model = model.fit(features_train,target_train)
pred_test = model.predict(features_test)
# transformar las predicciones en un DataFrame y guardarlo
submission = pd.DataFrame({'pos':pred_test})
print(submission)

# Resultado
#       pos
# 0       0
# 1       1
# 2       1
# 3       1
# 4       0
# ...   ...
# 2215    0
# 2216    1
# 2217    1
# 2218    1
# 2219    1
# [2220 rows x 1 columns]

#---------------------------- Insertados de palabras ----------------------------
# recientemente se toma en cuenta el contexto y el significado de las palabras al transformar el texto en vectores.
# Esto no es posible con el modelo "bolsa de palabras" o TF-IDF
# ambos modelos tratan cada palabra como un componente separado del texto.

# Las máquinas no pueden trabajar directamente con palabras, imágenes y audio
# Los modelos de lenguaje transforman el texto en vectores y utilizan un conjunto de técnicas denominadas "insertados de palabras"

# significa que una palabra está embebida en un espacio vectorial que representa un modelo de lenguaje. 

# Diferentes áreas en ese espacio vectorial tienen diferentes significados desde la perspectiva del lenguaje.
# Un vector creado a partir de una palabra mediante el insertado de palabras llevará información contextual sobre la palabra y sus propiedades semánticas. 
# De esta forma, no se pierde la definición convencional de una palabra y su significado en contexto.

# una palabra puede tener diferentes significados según el contexto.
# Esto significa que las palabras con contextos similares tendrán vectores similares.

# la similitud de ambos pares de palabras se define por lo bien que estas pueden encajar en el contexto. 
# Es más probable que los vectores cercanos encajen en las mismas oraciones.

# no siempre podemos interpretar los vectores como características.

#---------------------------- ¿Qué significa "embebida"? ----------------------------
# "Embebida" significa que la palabra se coloca (se "incrusta") dentro de un espacio matemático lleno de números.

# Analogía simple(mapa de una ciudad):
# Cada lugar tiene coordenadas (números) que indican dónde está
# Lugares similares (como restaurantes) están cerca unos de otros
# Lugares diferentes (como hospitales y parques) están más separados

# En el caso de las palabras:
# Cada palabra se convierte en un vector (una lista de números)
# Palabras con significados similares tienen vectores parecidos
# Palabras diferentes tienen vectores más distantes

# "Embebida" significa que la palabra se transforma en números de una manera inteligente que preserva su significado y contexto.

#---------------------------- Word2vec (palabra a vector) ----------------------------
# Word2vec es un método que se utiliza para convertir palabras en vectores
# para que las palabras cercanas semánticamente acerquen los vectores entre sí

# Las librerías que ofrecen este método suelen contener modelos word2vec entrenados previamente, con cada modelo entrenado en un corpus específico de textos.
# para obtener vectores relevantes para tus textos, debes seleccionar un modelo word2vec que se haya creado para un corpus de textos semánticamente relevante
# https://spacy.io/models/en

# otra librería popular para implementar el método word2vec
# https://radimrehurek.com/gensim/index.html
# https://github.com/piskvorky/gensim-data

# pueden ayudar a predecir si algunas palabras son vecinas o no. 
# Las palabras se consideran vecinas si caen en la misma "ventana" (distancia máxima entre palabras).

# Dado que cada palabra en un par de palabras es una característica, esto significa que hay dos características en cada par.

# Ejemplo: 
# texto original
# Red beak of the puffin flashed in the blue sky

# después de la lematización
# red beak puffin flash in blue sky

# las palabras vecinas de puffin son las más cercanas de ambos lados: "red", "beak", "flash" y "in"(cincograma)

# pares de vecinas
# puffin    red
# puffin    beak
# puffin    flash
# puffin    in

# Si tuviéramos que predecir la palabra "flash", las palabras vecinas serían: "beak," "puffin," "in," and "blue".

# cuatro nuevos pares de palabras vecinas:
# puffin    red
# puffin    beak
# puffin    flash
# puffin    in
# flash    beak
# flash    puffin
# flash    in
# flash    blue

# Podemos "desplazar la ventana" a través de todo el texto y extraer la lista completa de palabras vecinas.
# word2vec es entrenar un modelo para distinguir pares de vecinas verdaderas de las aleatorias
# es como una tarea de clasificación binaria
# las características son palabras 
# el objetivo es la respuesta a la pregunta de si las palabras son vecinas verdaderas o no.

# para obtener los negativos necesitaremos tomar palabras aleatorias del corpus y emparejarlas
# "1" indica que las palabras son verdaderas vecinas, mientras que "0" significa que no son vecinas.

# Durante el entrenamiento, seleccionamos los vectores de palabras que reflejan palabras vecinas.
# Reducimos la tarea de buscar vectores de palabras a una tarea de clasificación para averiguar si las palabras de los pares son vecinas o no.
# Tomamos en cuenta el significado de las palabras cuando creamos vectores basados en el contexto de intercambio de palabras.
# Formamos los pares de palabras a partir de los n-gramas del texto.

# Para que el modelo sea capaz de diferenciar los ejemplos positivos de los negativos necesitamos ejemplos negativos

# ¿Cómo encontramos los pares de palabras vecinas para entrenar word2vec?
# Encontramos los n-gramas a lo largo de todo el corpus de textos.
# el valor n determina la distancia máxima entre palabras vecinas.

#---------------------------- Insertados para clasificación ----------------------------
# la representación vectorial puede ayudar a resolver tareas de clasificación y regresión.

# tenemos un corpus de texto que necesita ser clasificado
# nuestro modelo constará de dos bloques:
#     1. Modelos para convertir palabras en vectores: las palabras se convierten en vectores numéricos.
#     2. Modelos de clasificación: se utilizan vectores como características.

# Proceso de Clasificación de Texto

# Entrada de datos:
# Textos de ejemplo → 
# 1. "I | go | to | island" (lista de token 1)
# 2. "I | find | my | peace" (lista de token 2)
# ...
# n. "absolute | love | be | everywhere" (lista de token n)
# ↓
# Vocabulario de tokens:
# Los tokens se procesan usando un vocabulario predefinido
# ↓
# Modelo para convertir palabras en vectores:
# - Lista de token 1 → Vector 1
# - Lista de token 2 → Vector 2
# - ...
# - Lista de token n → Vector n
# ↓
# Modelo de clasificación usando vectores como características:
# - Vector 1 → Predicción 1 (resultado: 1)
# - Vector 2 → Predicción 2 (resultado: 0)
# - ...
# - Vector n → Predicción n (resultado: 1)

# Explicación del flujo:
# El proceso sigue esta secuencia:
# Tokenización → Vectorización → Clasificación → Predicción

# Antes de pasar a la vectorización de palabras, necesitaremos realizar:
    # 1. Cada texto está tokenizado (descompuesto en palabras).
    # 2. Luego se lematizan las palabras (reducidas a su forma raíz). algunos modelos complejos, no requieren este paso porque entienden las formas de las palabras.
    # 3. El texto se limpia de palabras vacías o caracteres innecesarios.
    # 4. Para algunos algoritmos se agregan tokens especiales para marcar el comienzo y el final de las oraciones.
    # 5. Cada texto adquiere su propia lista de tokens después del preprocesamiento.
    # 6. Luego, los tokens se pasan al modelo, que los vectoriza mediante el uso de un vocabulario de tokens precompilado.
    #     En la salida obtenemos vectores de longitud predeterminada formados para cada texto.
    # 7. El paso final es pasar las características (vectores) al modelo. Luego, el modelo predice la tonalidad del texto: "0" — negativo o "1" — positivo.

#---------------------------- BERT (Representaciones de Codificador Bidireccional de Transformadores) ----------------------------
# es un modelo de red neuronal creado para la representación del lenguaje.
# Este algoritmo es capaz de "comprender" el contexto de un texto completo, no solo de frases cortas. 
# BERT se usa con frecuencia en el machine learning para convertir textos en vectores. 
# Los modelos más precisos actualmente son BERT y GPT.

# Al procesar palabras, BERT considera tanto las palabras vecinas inmediatas como las palabras más lejanas. 
# Esto permite que BERT produzca vectores precisos con respecto al significado natural de las palabras.

# como funciona:
# "The red beak of the puffin [MASK] in the blue [MASK] "
# MASK representa palabras desconocidas o enmascaradas. El modelo tiene que adivinar cuáles son estas palabras
# El modelo aprende a averiguar si las palabras del enunciado están relacionadas. 
# el modelo tiene que comprender que una palabra sigue a la otra.

# si escondieramos ciertas palabras en un texto, el modelo BERT podría predecir cuáles son esas palabras basándose en el contexto de las palabras vecinas.
# o podria fallar al crear una conexión entre palabras que no están relacionadas, lo que resultaría en una predicción incorrecta.

#---------------------------- BERT y preprocesamiento ----------------------------
import numpy as np
import torch #se utiliza para trabajar con modelos de redes neuronales
import transformers #implementa BERT y otros modelos de representación del lenguaje

# Antes de convertir textos en vectores, necesitamos preprocesar el texto. 
# BERT tiene su propio tokenizador basado en el corpus en el que fue entrenado. 
# Otros tokenizadores no funcionan con BERT y no requieren lematización

#inicializa el tokenizador como una instancia de BertTokenizer() con el nombre del modelo previamente entrenado.
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
example = 'Es muy práctico utilizar transformadores'
ids = tokenizer.encode(example, add_special_tokens=True)
#Convierte el texto en ID de tokens y el tokenizador BERT devolverá los ID de tokens en lugar de los tokens
print(ids)
#Resultado: [101, 2009, 2003, 2200, 18801, 2000, 2224, 19081, 102]

# los identificadores de token anteriores son esencialmente índices numéricos para tokens en el diccionario interno utilizado por BERT
# el diccionario se usó para entrenar BERT previamente, es parte del modelo BERT y está cargado con el método from_pretrained
# add_special_tokens = True: agregamos el token inicial (101) y el token final (102) a cualquier texto que se esté transformando
# BERT acepta vectores de una longitud fija, por ejemplo, de 512 tokens

# Si no hay suficientes palabras en una cadena de entrada para completar todo el vector con id tokens  el final del vector se rellena con ceros. 

# Si hay demasiadas palabras y la longitud del vector excede 510 (se reservan dos posiciones para los tokens de inicio y finalización), la cadena de entrada se limita al 510, o se suelen omitir algunos identificadores devueltos por tokenizer.encode()
#     todos los identificadores después de la posición 512 
n = 512
padded = np.array(ids[:n] + [0]*(n - len(ids))) 
print(padded)
# resultado: [101 2009 2003 2200 18801 2000 2224 19081 102 0 0 0 0 ... 0 ]

# tenemos que decirle al modelo por qué los ceros no tienen información significativa - componente attention
# Vamos a descartar estos tokens y crear una máscara para los tokens importantes, indicando valores cero y distintos de cero
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask.shape)
# resultado: (512, )

# Podemos establecer manualmente la longitud máxima de la entrada
# establece el parámetro max_length en el valor deseado y especifica truncation=True. Esto cortará los tokens que excedan el límite
# la parte resultante del texto inicial puede ser incluso más pequeña si add_special_token es True
example = 'It is very handy to use transformers'
ids = tokenizer.encode(example, add_special_tokens=True)
print(ids)
# resultado: [101, 2009, 2003, 2200, 18801, 2000, 2224, 19081, 102]

# truncado;
example = 'It is very handy to use transformers'
ids = tokenizer.encode(example, add_special_tokens=True, 
               max_length=5, truncation=True)
print(ids)
# resultado: [101, 2009, 2003, 2200, 102]
# 5 tokens, pero solo hay tres palabras presentes: max_length - tokens especiales = 5 - 2 = 3.

#---------------------------- clasificacion con insertados Ejemplo con BERT ----------------------------
# 1. Importar las librerías necesarias:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
#2. Cargar y preparar los datos:
df = pd.read_csv('imdb_reviews_200.tsv', sep='\t')
# Extraer las características (texto) y el objetivo
texts = df['review'].tolist()  #nombre de columna que contenga el texto
    # df['review']  # Esto selecciona toda la columna llamada 'review' del DataFrame
    # tolist()  # Convierte la Serie de pandas en una lista normal de Python
    # trabajas con pandas, las columnas son objetos tipo "Serie". Pero para procesarlas con BERT, necesitamos una lista simple de strings

    # Antes (Serie de pandas):
    # 0    "This movie was amazing..."
    # 1    "Terrible film, waste of time..."
    # 2    "Great acting and plot..."
    # Name: review, dtype: object

    # Después (lista de Python):
    # ["This movie was amazing...", 
    #  "Terrible film, waste of time...", 
    #  "Great acting and plot..."]
target = df['pos'].values  # variable objetivo 
    # Toma la columna 'pos' del DataFrame que contiene las etiquetas (positivo/negativo) 
    # El .values convierte la serie de pandas en un array de NumPy
print(f"Número de reseñas: {len(texts)}")
print(f"Distribución de clases: {np.bincount(target)}")
    # Esta función cuenta cuántas veces aparece cada valor en el array
    # resultado: [100, 100] significa que hay 100 reseñas negativas (0) y 100 reseñas positivas (1)
    # Si tienes números similares (como 100 y 100) → dataset balanceado 
    # Si tienes números muy diferentes (como 20 y 180) → dataset desbalanceado
#3. Inicializar BERT - Cargar el tokenizador y modelo de BERT:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# es el preprocesador de texto. Su trabajo es:
    # Convertir texto en números (IDs de tokens) que el modelo pueda entender
    # Dividir las palabras en subpalabras según el vocabulario de BERT
    # Agregar tokens especiales como [CLS] y [SEP]
    # Crear máscaras de atención
model = BertModel.from_pretrained('bert-base-uncased')
# El modelo es la red neuronal que:
#     Toma los IDs de tokens del tokenizador
#     Procesa estos números a través de múltiples capas transformer
#     Genera los embeddings (representaciones vectoriales) finales
model.eval()  # Modo evaluación
# Esta línea pone el modelo de BERT en modo evaluación (en lugar de modo entrenamiento).
#4. Función para obtener embeddings:
def get_bert_embeddings(texts, tokenizer, model, max_length=512):
    embeddings = []
    for text in texts:
        # Tokenizar el texto, Esta línea toma texto sin procesar y lo convierte en el formato que BERT necesita para entender y procesar el texto.
        inputs = tokenizer(text, 
            return_tensors='pt', #Le dice al tokenizador que devuelva los resultados como tensores de PyTorch (el formato que necesita el modelo BERT)
            max_length=max_length, 
            truncation=True, #Si tu texto tiene más tokens que max_length, se cortarán los tokens extras
            padding='max_length') #Rellena secuencias cortas hasta alcanzar max_length
        #  extrae la representación vectorial (embedding) de cada texto usando BERT.
        with torch.no_grad():
            # Le dice a PyTorch que NO calcule gradientes
            # Los gradientes solo se necesitan durante el entrenamiento
            # Como solo queremos extraer embeddings (no entrenar)
            outputs = model(**inputs)
            # Pasa los tokens procesados por el tokenizador al modelo BERT
            # El **inputs desempaqueta el diccionario (input_ids, attention_mask, etc.)
            # BERT procesa el texto y devuelve múltiples capas de representaciones
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            # Usar el token [CLS] como representación de la oración
            # last_hidden_state: La última capa de BERT (la más "refinada")
            # [:, 0, :]: Toma solo el primer token de cada secuencia
            # ¿Por qué el primer token? Porque es el token [CLS] que BERT usa como representación de toda la oración
            # .numpy(): Convierte de tensor PyTorch a array NumPy
            embeddings.append(cls_embedding.flatten())
            # flatten(): Convierte el embedding 2D en un vector 1D
            # append(): Añade este embedding a la lista de todos los embeddings
    return np.array(embeddings)
# 5. Generar embeddings y dividir datos:
print("Generando embeddings de BERT...")
features = get_bert_embeddings(texts, tokenizer, model)
# Dividir en entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.5, random_state=42
)
print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
#6. Entrenar el modelo y evaluar:
print("Entrenando modelo de regresión logística...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
# Calcular exactitud en entrenamiento
train_predictions = lr_model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Exactitud en conjunto de entrenamiento: {train_accuracy:.4f}")
# Opcional: evaluar en test también
test_predictions = lr_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Exactitud en conjunto de prueba: {test_accuracy:.4f}")

# El tokenizador es como el chef que prepara los ingredientes (corta, pela, organiza)
# El modelo es como el horno que cocina los ingredientes preparados

# Los modelos de deep learning tienen dos modos diferentes:
# 1. Modo entrenamiento (model.train()):
#     - Activa dropout (desconecta neuronas aleatoriamente)
#     - Actualiza batch normalization
#     - Usado cuando estás entrenando el modelo

# 2. Modo evaluación (model.eval()):
#     - Desactiva dropout (usa todas las neuronas)
#     - Congela batch normalization
#     - Usado cuando solo quieres hacer predicciones

# Como tú NO estás entrenando BERT, solo lo usas para extraer embeddings, necesitas ponerlo en modo evaluación para:
#     Obtener resultados consistentes y reproducibles
#     Usar toda la capacidad del modelo (sin dropout)
#     Evitar cambios internos no deseados

# Flujo 
# 1. Texto original
texto = "Esta película es genial"
# 2. Tokenizador convierte texto → números
ids = tokenizer.encode(texto)  # [101, 2023, 3185, 2003, 6581, 102]
# 3. Modelo convierte números → embeddings
embeddings = model(torch.tensor([ids]))