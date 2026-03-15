# Ejercicio 1: Cargar y visualizar el dataset MNIST
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def read_idx(filename): #Leer archivos IDX y convertirlos en arrays de NumPy.
    with open(filename, 'rb') as f: #Abrir el archivo en modo binario
        zero, data_type, dims = struct.unpack('>HBB', f.read(4)) #Leer los primeros 4 bytes para obtener el tipo de datos y la cantidad de dimensiones
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims)) #Leer el tamaño de cada dimensión, creando una tupla con la forma del array
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape) #lee todo el resto del archivo, Luego numpy lo convierte en un array
            #el resultado es: (num_imagenes, alto, ancho)
def load_data(path_imgs, path_labels): 
    X = read_idx(path_imgs)
    y = read_idx(path_labels)
    X= X.astype('float32') #volver a decimales
        # Las redes neuronales trabajan mejor con números decimales que con enteros.
    X /= 255 #normalizar los valores de píxeles al rango [0, 1]
        # Las redes neuronales aprenden mejor cuando los datos están en el rango 0-1 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return X_train, y_train, X_test, y_test
imgs = 'archivos/train-images.idx3-ubyte'
labels = 'archivos/train-labels.idx1-ubyte'
X_train, y_train, X_test, y_test = load_data(imgs, labels)
print('Dataset shape:', X_train.shape[0], "[n_images,hight,width]")
plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()

# Ejercicio 2: Generar una red neuronal multicapa
from tensorflow.keras.models import Sequential #tipo de modelo donde las capas se apilan una después de otra
from tensorflow.keras.layers import Dense, Flatten
    # Dense: Capa totalmente conectada, cada neurona se conecta con todas las neuronas de la capa anterior
    # Flatten: Convierte matrices en vectores.
from tensorflow.keras.optimizers import Adam
    # Ajusta los pesos de la red para minimizar el error, uno de los más usados en deep learning.
def create_model(input_shape, num_classes): #función para crear el modelo
    model = Sequential() #definir el modelo como secuencial
    model.add(Flatten(input_shape=input_shape)) #aplanar la imagen de entrada (28, 28) a un vector de 784 elementos
    model.add(Dense(64, activation='relu')) #numero de neuronas / funcion de activacion 
    # (relu es una función de activación que introduce no linealidad, permitiendo a la red aprender patrones complejos)
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Capa de salida
    # Número de neuronas = número de clases / Softmax convierte los valores en probabilidades, la más alta es la predicción
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  #prepara el modelo para entrenar:
                  #Usa el optimizador Adam con una tasa de aprendizaje de 0.001,
                  loss='sparse_categorical_crossentropy',
                  #La funcion de perdida mide qué tan equivocada está la red
                  metrics=['acc'])
                  #Define qué medir durante entrenamiento y evaluación (accuracy en este caso)
    return model
create_model((28, 28), 10).summary() #Forma de los datos de entrada / Número de clases a predecir / Muestra la arquitectura.

# Ejercicio 3: Entrenar un modelo de red neuronal utilizando datos de entrenamiento y evaluar con datos de prueba.
def train_model(model,train_data, test_data, batch_size=32, epochs=5):
    # batch_size: el tamaño del lote para el entrenamiento (de forma predeterminada es 32).
    history = model.fit(train_data[0],train_data[1], #independientes
                    #train_data es una tupla que contiene dos elementos: train_data = (X_train, y_train)
                    # Posición 0: Las características (imágenes)
                    # Posición 1: Las etiquetas (clases)
                    validation_data=test_data, #en tupla
                    batch_size=batch_size, 
                    epochs=epochs,
                    shuffle=True, #Mezlar los datos de entrenamiento
                    verbose=2)
    return history
imgs = 'archivos/train-images.idx3-ubyte'
labels = 'archivos/train-labels.idx1-ubyte'
X_train, y_train, X_test, y_test = load_data(imgs, labels)
model = create_model((28, 28), 10)
model_trained = train_model(model, 
                        (X_train,y_train),
                        (X_test,y_test),
                        batch_size=32,
                        epochs=10,
                        verbose=1)