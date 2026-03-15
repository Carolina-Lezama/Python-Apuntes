#Librerias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parte 1. Carga de datos
def load_data(path):
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.)
    data_datagen_flow = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)
    return data_datagen_flow

# Paso 2. Creacion del modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                  metrics=['acc'])
    return model

# Paso 3. Entrenamiento del modelo
def train_model(model, train_data, test_data, batch_size=None, epochs=10):
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              verbose=2)
    return model
input_shape = (150, 150, 3)
train = load_data('/datasets/fruits_train/')
test = load_data('/datasets/fruits_test/')
model = create_model(input_shape)
model = train_model(model, train, test)

# Salida esperada
# Epoch 1/10
# 52/52 - 5s - 92ms/step - acc: 0.1205 - loss: 2.3190 - val_acc: 0.1277 - val_loss: 2.2286
# Epoch 2/10
# 52/52 - 4s - 77ms/step - acc: 0.1277 - loss: 2.1508 - val_acc: 0.1787 - val_loss: 2.0814
# Epoch 3/10
# 52/52 - 4s - 77ms/step - acc: 0.2253 - loss: 2.0205 - val_acc: 0.2723 - val_loss: 1.9919
# Epoch 4/10
# 52/52 - 4s - 76ms/step - acc: 0.3398 - loss: 1.7431 - val_acc: 0.3362 - val_loss: 1.7097
# Epoch 5/10
# 52/52 - 4s - 76ms/step - acc: 0.4193 - loss: 1.5551 - val_acc: 0.4468 - val_loss: 1.6068
# Epoch 6/10
# 52/52 - 4s - 77ms/step - acc: 0.4988 - loss: 1.4020 - val_acc: 0.5064 - val_loss: 1.4600
# Epoch 7/10
# 52/52 - 4s - 78ms/step - acc: 0.5313 - loss: 1.3193 - val_acc: 0.5617 - val_loss: 1.3573
# Epoch 8/10
# 52/52 - 4s - 79ms/step - acc: 0.5590 - loss: 1.2239 - val_acc: 0.4851 - val_loss: 1.3992
# Epoch 9/10
# 52/52 - 4s - 77ms/step - acc: 0.5747 - loss: 1.1717 - val_acc: 0.5787 - val_loss: 1.3021
# Epoch 10/10
# 52/52 - 4s - 76ms/step - acc: 0.6036 - loss: 1.1210 - val_acc: 0.5489 - val_loss: 1.2353