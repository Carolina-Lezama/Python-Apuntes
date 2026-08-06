# Interfaz de línea de comandos (CLI)

Todas las instrucciones se transmiten escribiendo texto directamente en una consola o terminal

### Característica:

- Interacción: Teclado (Texto)
- Uso de Recursos: Muy ligero (RAM y CPU mínimos)
- Velocidad de Tareas: Muy rápida para tareas repetitivas/masivas
- Automatización: Alta (mediante scripts o archivos batch)
- Curva de Aprendizaje: Alta (requiere memorizar comandos)

# Interfas gráfica de usuario (GUI)

La interfaz gráfica tradicional con ventanas, botones e íconos que se manipulan con el ratón.

### Característica:

- Interacción: "Ratón, gestos, clics"
- Uso de Recursos: Más pesado (consume recursos gráficos)
- Velocidad de Tareas: Más lenta si requiere muchos clics o carpetas
- Automatización: Limitada a macros o herramientas externas
- Curva de Aprendizaje: Baja (muy intuitiva y visual)

# Abrir la terminal

1. Presiona Win + R para abrir la ventana Ejecutar
2. Cada vez que quieras abrir Linux desde Windows, dirigete a PowerShell y escribir wsl. Mientras mantengas esta ventana abierta, Linux estará corriendo en paralelo en tu computadora junto a Windows.
3. Ctrl + Alt + T o buscando "terminal" en el Dash.

# Estructura de los comandos

comando -opciones del comando -argumentos del comando

# Opciones del comando

Modifican el comportamiento del comando y se indican con un guion (-) o dos guiones (--) antes del nombre de la opción.
Si no se dan se usaran valores predeterminados.
Puedes incluir una o varias opciones después del comando.

# Argumentos del comando

Elementos sobre los que actúa el comando y se escriben después del comando y sus opciones.
Pueden aceptar uno o varios argumentos, que son datos adicionales sobre los que el comando actuará.

### Muestra los archivos en orden inverso.

ls -r

### Muestra todos los archivos, incluidos los ocultos, en orden inverso.

ls -r -a
ls -ra

Los nombres de todos los archivos y directorios ocultos comienzan siempre por .

### Muestra los archivos y directorios dentro del directorio pasado como parametro en orden inverso

ls -r sql_project

### Mostrar un mensaje de ayuda detallado sobre ese comando en particular.

ls --help

### Abre el manual de usuario del comando ls

man ls

pwd responde a las siglas en ingles "print working directory": el comando pwd te muestra la ruta completa del directorio en el que te encuentras actualmente en la línea de comandos.

### Moverse a una carpeta

cd ./intro_to_ml/

### Nos llevaría directamente a Tu carpeta personal

cd ~

clear no borra el historial de los comandos que has ejecutado. Simplemente limpia la visualización en la pantalla.
Importante: Ten en cuenta que este historial de comandos se guarda solo mientras la ventana de la Terminal esté abierta.

Si presionas la tecla Tabulador dos veces rápidamente, la Terminal te mostrará una lista de todos los comandos que comienzan con "pw":

### Crea un nuevo directorio en el directorio actual.

mkdir mi_nueva_carpeta

### Multiples directorios

mkdir carpeta_uno carpeta_dos carpeta_tres

### Creando carpetas dentro de carpetas (subdirectorios); solo funcionará si la carpeta principal ya existe

mkdir mi_carpeta_principal/mi_subcarpeta

### p sirve para crear ambas si ninguna existe

mkdir -p mi_carpeta_principal/mi_subcarpeta

### Mostrar cada accion que se esta realizando

mkdir -p -v mi_carpeta_principal/mi_subcarpeta

### Eliminar directorios vacios

~$ rm -d mi_carpeta_temporal
~$ rm --dir mi_carpeta_temporal

### Pedir confirmacion

~$ rm -d -i mi_carpeta_temporal

### Comando echo: mostrar texto en la pantalla

~$ echo Hola, mundo de la CLI  
~$ echo -e 'Quiero aprender sobre la CLI.\n¡Todo programador la conoce!'

-e habilita los caracteres de escape

echo "El inicio de una gran aventura..." > aventura_parte1.txt
echo "Y así continuó el viaje." > aventura_parte2.txt

~$ echo "Aprendiendo comandos básicos de la CLI" > mi_primer_archivo.txt

Enviar su salida directamente a un archivo de texto. Si el archivo.txt no existe, lo creará automáticamente. Si ya existe, el símbolo > sobrescribirá todo su contenido con el nuevo texto.

~$ echo "¡Este es un nuevo renglón!" >> mi_primer_archivo.txt

Agregar nuevo texto al final de un archivo existente sin borrar su contenido

### Comando cat: mostrar el contenido de un archivo de texto en la pantalla

~$ cat mi_primer_archivo.txt  
cat importante.txt
cat aventura_parte1.txt aventura_parte2.txt

cat > m_nota.txt

crear un archivo de texto y escribir en él, escribe el texto que quieras y cuando termines, presiona Ctrl + D para guardar y salir

cat importante.txt > copia_importante.txt copiar el contenido de un archivo a otro

### Comando touch

crea archivos vacíos de forma instantánea. A estos se les llama "archivos dummy" porque existen, pero no contienen ningún dato.

~$ touch mi_archivo_vacio.txt

### Comando mv: mover la ubicacion del archivo

~$ touch mi_documento.txt  
~$ mkdir mis_documentos
~$ mv mi_documento.txt mis_documentos

~$ mkdir ~/archivos_importantes
~$ mv mi_documento.txt ~/archivos_importantes

~$ touch archivo_uno.txt archivo_dos.txt archivo_tres.txt
~$ mkdir carpeta_destino
~$ mv archivo_uno.txt archivo_dos.txt archivo_tres.txt carpeta_destino

mover carpetas

~$ mkdir mi_carpeta_antigua mis_archivos  
~$ mv mi_carpeta_antigua mis_archivos

cambiar el nombre de un archivo, al renombrar un archivo a un nombre que ya existe, ya que el archivo existente será reemplazado

~$ touch documento_viejo.txt
~$ mv documento_viejo.txt documento_nuevo.txt

### Comando cp(ubicacion y nombre): copiar carpeta con documentos dentro

~$ touch informe_original.txt
~$ cp informe_original.txt copia_informe.txt

~$ mkdir ~/respaldo_documentos  
~$ cp -r mis_documentos ~/respaldo_documentos
