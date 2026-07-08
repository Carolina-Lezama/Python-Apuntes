# Arbol de decisión - Decision Tree

Funcionan como un diagrama de flujo, toma decisiones complejas rompiendolas en series de preguntas basadas en los datos.

## Componentes

- Nodo Raíz (Root Node): primera pregunta del arbol, caracteristica mas importante
- Nodos de Decisión o Internos (Split Nodes): preguntas intermediarias, se llaman bifurcacion o subpregunta
- Nodos Hojas (Leaf Nodes): extremos finales del árbol. Ya no contienen preguntas, sino la respuesta o predicción final

## ¿Cómo sabe el algoritmo qué preguntas hacer primero?

Tú no tienes que inventar las preguntas; el algoritmo analiza y calcula matemáticamente cuál es la mejor variable para empezar a dividir los grupos.

1. Entropía / Ganancia de Información: grupo. El algoritmo busca preguntas que separen los datos en grupos lo más puros posibles (por un lado puros "SÍ" y por el otro puros "NO").
2. Impureza de Gini: mide la probabilidad de clasificar incorrectamente un elemento si lo eligieras al azar. El algoritmo quiere que la impureza llegue a cero.

## Tipos de Árboles de Decisión

- Árboles de Clasificación: Cuando la respuesta final es una categoría
- Árboles de Regresión: Cuando la respuesta final es un número continuo

## Problemas

Los árboles de decisión son "tercos". Si los dejas crecer sin control, harán preguntas tan específicas para ajustarse a tus datos de entrenamiento que memorizarán el dataset (Overfitting o sobreajuste).

En el análisis de datos, las filas se llaman instancias, mientras que las columnas son las variables.

En el machine learning, las filas y columnas representan observaciones
y características, respectivamente.

# Ramdon Forests

Este algoritmo entrena una gran cantidad de árboles independientes y toma una decisión mediante el voto. Un bosque aleatorio ayuda a mejorar los resultados y a evitar el sobreajuste.

Obtener una valoración promedio que anule el sesgo personal y los errores.

Usaremos el hiperparámetro n_estimators para establecer el número de árboles en el bosque.

El aumento en la cantidad de estimadores siempre disminuye la varianza de la predicción, por lo que cuantos más árboles uses, mejores resultados obtendrás.

Los bosques no pueden sobreajustarse debido a que tienen demasiados árboles. Si bien el sobreajuste de un bosque aún puede ocurrir debido al sobreajuste de sus árboles individuales, este efecto generalmente se ve compensado por el beneficio de tener muchos árboles.

El uso de más y más árboles incurre en un costo computacional cada vez mayor y sufre de rendimientos decrecientes.

Eventualmente, la métrica de calidad del modelo alcanza una meseta y deja de mejorar, mientras que el tiempo de ejecución sigue aumentando.

