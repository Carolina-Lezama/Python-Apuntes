# Verificar qué versiones de Python tienes instaladas

Comando:
py -0

Qué hace:
Muestra todas las versiones de Python instaladas en tu sistema.

Ejemplo de salida:
-3.12
-3.11
-3.10

Ciertas librerias ocupan ciertas versiones de Python, como TensorFlow.

# Crear un entorno virtual

Comando:
py -3.10 -m venv [nombre_que_quieras]
py -3.10 -m venv .venv # Nombre común (oculto)
py -3.10 -m venv mi_proyecto # Nombre descriptivo
py -3.10 -m venv env # Nombre corto
py -3.10 -m venv tensorflow_env # Nombre específico
py -3.10 -m venv frutas_cnn # Relacionado con tu proyecto

Explicación completa de cada parte:

- py Lanza el Python Launcher de Windows
- -3.10 Indica que se usará Python 3.10
- -m Ejecuta un módulo de Python
- venv Módulo que crea entornos virtuales
- .venv Nombre de la carpeta del entorno

Crea una carpeta que contiene un Python aislado solo para tu proyecto.
Se usa para evitar conflictos entre proyectos.

Ejemplo:
Proyecto A:
TensorFlow 2.13
numpy 1.24

Proyecto B:
PyTorch
numpy 2.0

Sin entornos virtuales se rompen entre sí.

# Activar el entorno virtual

Comando:
.\[nombre_de_carpeta]\Scripts\activate
.\.venv\Scripts\activate

Explicación de cada parte:

- .\ ruta actual
- .venv carpeta del entorno
- Scripts carpeta donde están los ejecutables
- activate script que activa el entorno

Cambia el Python activo de la terminal al del entorno.

Antes:
C:\Python312\python.exe

Después:
.\.venv\Scripts\python.exe

En la terminal aparecerá:
([nombre_de_carpeta]) indicando que el entorno está activo.

# Actualizar pip (opcional pero recomendado)

Comando:
py -m pip install --upgrade pip

Actualiza pip, el gestor de paquetes de Python.

# Instalar TensorFlow

Comando:
pip install tensorflow

# Instalar Pillow

Comando:
pip install pillow

# Verificar instalación

Comando:
py

Luego dentro de Python:

import tensorflow as tf
print(tf.**version**)

Salida esperada:
2.16.x

Para salir:
exit()

# Ejecutar el script

Comando:
py ejercicio_1.py

Ejecuta el script con el Python del entorno virtual.

# Guardar dependencias del proyecto

Comando:
pip freeze > requirements.txt

Guarda todas las librerías instaladas.

Ejemplo:
tensorflow==2.16.1
numpy==1.26.4
pillow==10.3.0
keras==3.2.1

# Reinstalar entorno en otro proyecto

Comandos:
py -3.10 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

Instala exactamente las mismas dependencias.

# Desactivar el entorno

Comando:
deactivate

Regresa al Python global del sistema.
