# Red neuronal convolucional entrenada con el conjunto de datos de frutas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Parte 1. Carga de datos
def load_data(path):
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
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345
    )
    val_datagen_flow = validation_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='validation',
        seed=12345
    )
    return train_datagen_flow, val_datagen_flow

# Paso 2. Creación del modelo
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(train_data.num_classes, activation='softmax')) #Automatico
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Paso 3. Entrenamiento
def train_model(model, train_data, val_data, epochs=10):
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        steps_per_epoch=200,
        validation_steps=50
    )

# Ejecución
input_shape = (150,150,3)
train_data, val_data = load_data('datos/fruits_small')
model = create_model(input_shape)
train_model(model, train_data, val_data)
print("Clases:", train_data.class_indices)
print("Numero de clases:", train_data.num_classes)

# Resultados obtenidos al ejecutar el código:
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1773612447.182179   18936 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1773612458.258483   18936 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Found 52917 images belonging to 15 classes.
# Found 17632 images belonging to 15 classes.
# C:\Users\michi\OneDrive\Desktop\Python\Ejercicios\clasificacion de frutas\.venv\lib\site-packages\keras\src\layers\convolutional\base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# I0000 00:00:1773612467.200675   18936 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
# Epoch 1/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 93s 455ms/step - accuracy: 0.2775 - loss: 2.4140 - val_accuracy: 0.2775 - val_loss: 2.2801
# Epoch 2/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 81s 406ms/step - accuracy: 0.3600 - loss: 2.0636 - val_accuracy: 0.3875 - val_loss: 1.8736
# Epoch 3/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 66s 334ms/step - accuracy: 0.4319 - loss: 1.7700 - val_accuracy: 0.4288 - val_loss: 1.7296
# Epoch 4/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 52s 260ms/step - accuracy: 0.4762 - loss: 1.6164 - val_accuracy: 0.4500 - val_loss: 1.5680
# Epoch 5/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 46s 230ms/step - accuracy: 0.5309 - loss: 1.4176 - val_accuracy: 0.4900 - val_loss: 1.4951
# Epoch 6/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 40s 203ms/step - accuracy: 0.5616 - loss: 1.3283 - val_accuracy: 0.5387 - val_loss: 1.3518
# Epoch 7/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 52s 264ms/step - accuracy: 0.5959 - loss: 1.1801 - val_accuracy: 0.5962 - val_loss: 1.2244
# Epoch 8/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 42s 208ms/step - accuracy: 0.6341 - loss: 1.0787 - val_accuracy: 0.6175 - val_loss: 1.1267
# Epoch 9/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 32s 162ms/step - accuracy: 0.6550 - loss: 1.0442 - val_accuracy: 0.6050 - val_loss: 1.1278
# Epoch 10/10
# 200/200 ━━━━━━━━━━━━━━━━━━━━ 30s 152ms/step - accuracy: 0.6931 - loss: 0.9248 - val_accuracy: 0.6338 - val_loss: 1.0348
# Clases: {'Apple': 0, 'Banana': 1, 'Carambola': 2, 'Guava': 3, 'Kiwi': 4, 'Mango': 5, 'Orange': 6, 'Peach': 7, 'Pear': 8, 'Persimmon': 9, 'Pitaya': 10, 'Plum': 11, 'Pomegranate': 12, 'Tomatoes': 13, 'muskmelon': 14}
# Numero de clases: 15
