# Guía de Git usada en este proyecto

### git status

## Qué hace

Muestra el estado actual del repositorio(verificar qué agregaste):

- archivos modificados
- archivos nuevos
- archivos listos para commit
- archivos ignorados

### git add

## Ejemplo de uso

git add archivo.py
git add .

## Qué hace

Agrega archivos al staging area, es decir, prepara los archivos que irán al próximo commit.
Después de modificar o crear archivos que quieres guardar en el repositorio.

### git commit

## Ejemplo

git commit -m "mensaje"
git commit -m "agregar scripts de clasificacion"

## Qué hace

Crea un snapshot del proyecto con los archivos que están en staging.

## Cuándo usarlo

Después de git add.

### git push

## Ejemplo

git push origin main

## Qué hace

Sube tus commits locales al repositorio remoto en GitHub.

## Estructura

git push [remoto] [rama]

### git push -f

## Ejemplo

git push -f origin main

## Qué hace

Fuerza la subida, reemplazando lo que está en el remoto con tu versión local.

## Cuándo usarlo

Cuando quieres reemplazar completamente el repositorio remoto o limpiar commits viejos

### git reset

## Ejemplo

git reset

## Qué hace

Quita archivos del staging area sin borrar cambios.

## Ejemplo

Si hiciste git add . puedes revertirlo con git reset

### git rm --cached

## Ejemplo

git rm -r --cached .

## Qué hace

Elimina archivos del control de Git sin borrarlos del disco.

## Para qué lo usamos

Para limpiar archivos como:

- .venv
- site-packages
- datasets

### .gitignore

## Ejemplo

Archivo que indica a Git qué no debe subir.

## Ejemplo para Python

- .venv/
- venv/
- \_.pkl
- **pycache**/
- \_.pyc
- .ipynb_checkpoints/
- \_.zip
- \_.csv

### pip freeze

## Ejemplo

pip freeze > requirements.txt

## Qué hace

Guarda todas las librerías instaladas del entorno.

## Para qué sirve

Permite recrear el entorno con:

- pip install -r requirements.txt

### Archivo de bloqueo de Git.

Solución:

- Remove-Item -Force .git\index.lock

repositorio muy pesado
