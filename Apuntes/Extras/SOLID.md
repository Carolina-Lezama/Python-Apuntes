# Principios SOLID

Cinco reglas de oro del diseño y desarrollo de software orientado a objetos, ideadas para crear código más limpio, fácil de mantener y escalable.

## S - Single Responsibility Principle 
Principio de Responsabilidad Única: 

Cada clase o módulo debe tener una única responsabilidad(funcionabilidad) y, por tanto, una única razón para cambiar.

## O - Open/Closed Principle
Principio de Abierto/Cerrado: 

Las entidades de software deben estar abiertas a la extensión, pero cerradas a la modificación. Esto permite añadir nuevas funciones sin alterar el código existente.

## L - Liskov Substitution Principle 
Principio de Sustitución de Liskov: 

Los objetos de un programa deben ser reemplazables por instancias de sus subtipos sin alterar el correcto funcionamiento del sistema.

De las clases padre y las clases hijas,  sus instancias deben poderse intercambiar sin producir resultados inesperados. Debemos evitar meter restricciones o cambiar el funcionamiento de los metodos heredados para no tener comportamientos inesperados

## I - Interface Segregation Principle 
Principio de Segregación de Interfaz: 

Es mejor tener muchas interfaces pequeñas y específicas para cada cliente, que una sola interfaz grande y genérica.

## D - Dependency Inversion Principle 
Principio de Inversión de Dependencia: 

Los módulos de alto nivel no deben depender de los módulos de bajo nivel. Ambos deben depender de abstracciones (como las interfaces).

En lugar de recibir dependencias, recibe dependencias atraves de la interfaz