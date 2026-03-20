# Obtén el valor EAM para el conjunto de pruebas no superior a 8

#Librerias
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#carga los conjuntos de datos de entrenamiento y prueba
def load_data(path, subset=None):
    labels = pd.read_csv(path + 'labels.csv')
    
    if subset:
        data_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    else:
        data_datagen = ImageDataGenerator(rescale=1./255)

    data_gen_flow = data_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset=subset,
        seed=12345)

    return data_gen_flow

# define un modelo
def create_model(input_shape):
    
    backbone = ResNet50(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )
    
    backbone.trainable = False  #sino estaria entrenando toda la red y seria muy lento

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5)) #para prevenir el overfitting, apaga aleatoriamente el 50% de las neuronas
    model.add(Dense(1))  #mejor para regresion, relu limita valores negativos (aunque edad no lo es, puede afectar gradientes)

    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model

#controla el entrenamiento del modelo   
def train_model(model, train_data, val_data, epochs=10):

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=2
    )

    return model


input_shape = (224, 224, 3)

train = load_data('/datasets/train/', subset='training')
val = load_data('/datasets/train/', subset='validation')
test = load_data('/datasets/test/')

model = create_model(input_shape)
model = train_model(model, train, val)

# evaluación final
model.evaluate(test)