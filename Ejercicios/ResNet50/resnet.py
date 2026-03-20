# usar ResNet50 para entrenar una clasificación de imágenes de frutas.

# 0. Importar las librerías necesarias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet import ResNet50

# 1. cargar datos
def load_data(path, width, height, batch_size):
    train_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
    )
    validation_datagen = ImageDataGenerator(
        validation_split=0.25,
        rescale=1./255
    )
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        seed=12345,
        shuffle=True, #Evita que el modelo aprenda patrones del orden de los datos, cada epoca ve datos diferentes 
    )
    val_datagen_flow = validation_datagen.flow_from_directory(
        path,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        seed=12345,
    )
    return train_datagen_flow, val_datagen_flow

# 2. Crear el modelo() 
def create_resnet_model(input_shape):
    backbone = ResNet50(
        input_shape=input_shape, 
        weights='imagenet', 
        include_top=False
        )
    # usaremos un modelo previamente entrenado con sus parámetros aprendidos como funciones para entrenar nuestro nuevo modelo
    for layer in backbone.layers:
        layer.trainable = False
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D()) # Paso 2: Aplanar datos
    model.add(Dense(train_gen.num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Entrenar el modelo()
def train_model(model, train_data, val_data, epochs=10):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch = len(train_data),
        validation_steps = len(val_data)
    )
    return model

# 4. usar el modelo:
## Parámetros
width = 160
height = 128
batch_size = 16
input_shape = (width, height,3)
## Datos almacenados
path = 'dataset/fruits/'
# Carga los datos de entrenamiento y validación
train_gen, val_gen = load_data(path, width, height, batch_size)
# Crea y entrena el modelo simple
model = create_resnet_model(input_shape)
trained_model = train_model(model, train_gen, val_gen, epochs=10)