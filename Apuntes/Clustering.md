# Agrupación en clústeres de K-means

Ejemplo común de método de clustering exclusivo; en el que los puntos de datos se asignan a K grupos, donde K representa el número de clusters basado en la distancia desde el centroide de cada grupo. Los puntos de datos más cercanos a un centroide determinado se agruparán en la misma categoría.

Un valor K mayor indica agrupaciones más pequeñas con más detalle, mientras que un valor K menor indica agrupaciones más grandes y menos detalle.

# Clustering

El modelo busca relaciones entre las observaciones. El análisis de clústeres (clustering) es la tarea de combinar observaciones similares en grupos o clústeres.

La mayoría de los métodos de clustering implican determinar la similitud o diferencia de las observaciones con base en la distancia entre ellas. Cuanto más lejos estén las observaciones entre sí, menos similares serán, y viceversa.

# Clustering de k-means

El concepto clave de este algoritmo es el centroide, o el centro de un clúster. El grado de cercanía a un centro en particular depende de a qué clúster pertenece la observación. Cada clúster tiene su propio centroide, que se calcula como la media aritmética de las observaciones agrupadas.

K-Means: toma datos que no tienen etiquetas y los divide en grupos basados en qué tan cerca están unos de otros. La "K" es el número de grupos (o clusters) que tú decides crear, y "Means" (medias) se refiere al centro geométrico de cada grupo.

Otra condición para detener el trabajo del algoritmo es el hiperparámetro para el máximo número de iteraciones (max_iter).

Consejo fundamental para principiantes sobre K-Means:

Escalar siempre los datos: K-Means calcula distancias matemáticas (como la distancia euclidiana) para agrupar los puntos. Si una variable está en miles y otra en unidades, la variable de ingresos va a dominar por completo el algoritmo.

Funciona asignando iterativamente cada dato al cluster cuyo centroide (media del grupo) esté más cercano, y luego recalculando estos centroides hasta que la asignación se estabilice.

**max_iter:** Indica el número máximo de iteraciones permitidas en cada ejecución del algoritmo. Si se alcanza este límite sin que la asignación de los datos cambie, el proceso se detiene.

Durante el algoritmo, estos centroides _se actualizan iterativamente para reflejar mejor la posición central de sus respectivos clusters_, ayudando a optimizar la agrupación de los datos.

