# Redes Neuronales WORK IN PROGRESS!
En este apartado se incluyen los conocimientos necesarios para entender el funcionamiento de las redes neuronales y su uso para problemas de segmentación.
## 1 Introducción a las redes neuronales
Una red neuronal es un sistema de procesamiento de la información compuesto por nodos o neuronas interconectadas. Estas neuronas asignan un valor numérico o peso a sus entradas y producen una salida que permite resolver problemas de regresión (predecir una cantidad) y clasificación (predecir una clase).
### 1.1 Funcionamiento de una neurona
Cuando las entradas a una neurona superan un umbral, esta se activa propagando su salida. Las conecciones de entrada que son fuertes aportan más que las débiles a esta activación. A continuación se describe el modelo matemático más simple que implementa el funcionamiento de una neurona biológica.
<img src="http://drive.google.com/uc?export=view&id=0B2_2YevJnwmmeGYyVi1kOGFBUDcwbFBHUnJHNW9HTzNHanhB" alt="Modelo de una neurona">

Siendo xi la entrada y los parámetros ajustables wi y b, tenemos que la función para la regresión lineal es:
z=i(wixi) + b
Si utilizamos una función para la activación de la neurona  que aplique un umbral sobre z, a = (z), obtenemos la ecuación de un clasificador lineal:
a = (i(wixi) + b) 

En el caso de que la función de activación sea (z)=1  para z>0 y (z)=0 para z≤0 la neurona puede resolver problemas de clasificación binaria. Utilizando dos entradas (x1,x2), dos tipos de datos (Rojo, Azul) y el siguiente hiperplano:

Siendo m la pendiente y c el punto de corte con el eje x2, tenemos que el espacio de activación (sombreado) viene viene descrito por:
 x2 < mx1 + c  mx1-x2+c >0  

Al compararlo con la ecuación de la neurona para la clasificación binaria w1x1+w2x2+b >0 obtenemos los valores w1=m, w2= -1 y b=c . 
De esta forma la neurona se activará siempre que la entrada se encuentre en la región descrita por sus parámetros internos.

La colocación de neuronas en forma de capas se denomina arquitectura multicapa y permite producir regiones de decisión de diferente complejidad y forma, siempre y cuando no se utilicen funciones de activación lineales. Al número de capas se le conoce por profundidad.


Arquitectura multicapa con dos salidas.
Siendo la salida de la neurona j en la capa L igual a:
ajL = jL(zjL) , zjL=i(wi,jLaiL-1) + bjL, yi0 =xi
Se puede convertir en operaciones con matrices:
AL = L(ZL), ZL=WLAL-1 + BL, Y0=X

Cuando usamos la función de activación identidad ((z)=z) en arquitecturas multicapa, al igual que el resto de funciones de activación lineales, se produce una salida lineal. Esto quiere decir que equivale a implementar una monocapa como se muestra en la siguiente transformación:

Utilizando una red de dos capas con una neurona en cada capa y una sola entrada:

 y=w2(w1x1+b1)+b2 
wmonocapa = w1w2 y bmonocapa=w2b1 + b2

Se comprueba que el resultado sigue siendo una función lineal, pudiendo reducirse a una sola neurona y perdiendo la capacidad de solucionar problemas complejos.

La introducción de funciones no lineales permiten a la redes neuronales, con los suficientes parámetros y tiempo, modelar cualquier tipo de función en arquitecturas multicapa. Es común utilizar un tipo de activación para construir la red y otro diferente justo antes de la salida para cambiar el tipo de resultado buscado (clasificación binaria, regresión…).


Ejemplo de espacio de decisión complejo



Un problema complejo es capaz de necesitar miles de neuronas y el ajuste de sus parámetros se vuelve una práctica difícil de hacer manualmente. Según el tipo de problema y la información conocida previamente, existen diferentes técnicas de aprendizaje que se pueden agrupar en 3 tipos.

Aprendizaje por refuerzo
Los datos de entrada no son conocidos directamente. Existe un agente (la red neuronal) que interacciona con un entorno o medio realizando acciones y cambiando de estado. Las acciones en cada paso que ayuden a llegar al estado deseado son reforzadas de manera que, cuando el agente se encuentre en la misma situación, seleccione la acción con mayor recompensa.

Aprendizaje no supervisado
Los datos de entrada son conocidos y utilizamos una función de coste dependiente de los valores de entrada y la salida de la red que tratamos de minimizar. La selección de esta función depende del problema a solucionar.

Aprendizaje supervisado
En el aprendizaje supervisado se conocen los pares de datos de entrada y la salida. Se utiliza una función de error, como puede ser el error cuadrático medio, entre la salida de la red y la salida deseada para ajustar los parámetros de la red.
### 1.2 El Gradiente y la propagación hacia atrás

El gradiente es un vector de n-dimensiones que proporciona la dirección de crecimiento y la variación de una función con respecto a los cambios de sus variables independientes. En nuestro caso el gradiente del error tendrá tantas dimensiones como variables entrenables y, al conocer hacia dónde decrece la función de error, podremos encontrar sus valores óptimos.

⛛E = Ew1, Eb1 ,...,EwL, EbL




Recordar que cuando tenemos una función dentro de otra,  f(g(x)), su derivada es la variación de f con respecto a g por la variación de g respecto a la variable independiente x. Esto se puede aplicar tantas veces como funciones intermedias existan por lo que es conocida como la regla de la cadena.

fx=fggccy...bx

En el caso de una arquitectura con una neurona por capa definimos su función de error como:
E(w1,b1,...,wL,bL) = (ypredicción - yesperada)2= (aL- yesperada)2
aL=(zL) y zL = wL aL-1 +bL 

El cálculo de las componentes del gradiente en cada capa viene dado por:


Capa de Salida L:
EwL=EaLaLwLaLwL=aLzLzLwL
EwL=EaLaLzLzLwLzLwL=aL-1,EaL= 2(aL-y),aLzL = '(zL)

En el caso del bias hay que sustituir zLwLzLbL=1

Capa L-1:
EwL-1=EaLaLwL-1aLwL-1=aLzLzLwL-1zLwL-1= zLaL-1aL-1wL-1 aL-1zL-1 zL-1wL-1
EwL-1=EaLaLzL zLaL-1aL-1zL-1zL-1wL-1zL-1wL-1=aL-2, zLaL-1= wL, aL-1zL-1='(zL-1)

Capa L-2:
EwL-2=EaL aLzLzLaL-1aL-1zL-1zL-1aL-2aL-2zL-2zL-2wL-2
zL-2wL-2=aL-3, zL-1aL-2= wL-1, aL-2zL-2='(zL-2)

Capa X:
EwX=2(aL-y) aX-1i=xi=L('(zi))i=x+1i=Lwi


Como se observa en las ecuaciones, todas las operaciones deben ser diferenciables y, para saber cuánto aportan los parámetros de una capa al error total, es necesario conocer el resultado de su variación por las capas siguientes (elementos coloreados). Para ello comenzamos el cálculo desde la salida dándosele el nombre de propagación hacia atrás del gradiente. 

Existen dos problemas con el cálculo del gradiente a medida que aumenta la profundidad de la red. Supongamos el siguiente caso.

Uno de los términos de los que depende EwX son los pesos de las capas consecuentesi=x+1i=Lwi.
Utilizando el mismo peso en cada capa, cuando |w|>1 obtenemos un aumento o explosión del gradiente pero si 0<|w|<1 tendemos a 0, lo que se conoce como desvanecimiento del gradiente. En caso de que sature, las capas más cercanas a la entrada de la red tendrán mayores modificaciones que las últimas. Si por el contrario ocurre el desvanecimiento, las modificaciones de los pesos son insignificantes haciendo que la red deje de aprender en las primeras capas y repercutiendo en el resto. 

La modificación de los pesos tambien depende de la función de activación elegida i=xi=L('(z))permitiendo mitigar o agravar los problemas de gradiente como veremos en el siguiente apartado. Otra opción bastante popular es añadir conexiones residuales que sumen la señal de entrada de una capa con su salida de forma que cada una afecta directamente al error de la red. aL = (x +a1 + a2 + … + aL-1)

Otras soluciones añaden un componente dentro del cálculo del error como las funciones L1 y L2 de regularización. Este tipo de funciones son dependientes de los pesos, penalizando los valores muy grandes o muy pequeños.

Conocidos todos los componentes del gradiente es hora de minimizar la función de error. Un método de optimización común se  conoce como descenso por gradiente que mediante saltos, proporcionales a la aportación del error, en la dirección de decrecimiento, resulta en un proceso iterativo para la búsqueda de mínimos. La cantidad de salto se denomina ratio de aprendizaje y valores muy grandes pueden provocar que no llegue a converger en el mínimo (zig-zag) o incluso divergir mientras que valores muy pequeños hace que el  aprendizaje sea más lento.

w = wanterior-Ew

No suele ser necesario decrecer el ratio de aprendizaje con el número de iteraciones ya que los saltos son proporcionales al módulo del gradiente y tiende a 0 a medida que nos acercamos a un mínimo.
 
Ejemplo de descenso por gradiente sobre la función de error para una variable.
En caso de que el tipo de datos de entrada tenga grandes diferencias entre sí puede ocurrir que la red deshaga los cambios en los pesos al tratar de minimizar sus errores por separado. Por lo tanto, se recomiendo calcular la media del error sobre un conjunto o lote de datos que preferiblemente contenga un caso de cada tipo. En numerosas ocasiones no es posible separar en diferentes clases los datos de entrada, por lo que suele recomendarse su reordenación de manera aleatoria antes de extraer el mayor número posible para el lote.

### 1.3 Funciones de activación

Como vimos anteriormente las funciones de activación juegan un papel importante en el cálculo del gradiente. Es común encontrar en una arquitectura multicapa dos tipos de funciones de activación, el que utiliza las capas internas para aprender y el que utiliza la capa de salida para producir el resultado del problema tratado.

A continuación nombramos cuatro de las funciones más comunes:

Sigmoid. La función sigmoidal o logística se basa en la activación de una neurona biológica que devuelve valores comprendidos entre 0 y 1. Observando su derivada, comprobamos que agrava el problema de desvanecimiento del gradiente en los extremos, donde tiende a cero.

Su valor a la salida de la red se interpreta como la probabilidad de pertenecer a una clase. Cuando se tienen múltiples salidas no tienen por qué sumar uno entre sí (80% gato y 95% animal). 

Tanh. Parecida a la sigmoidal pero entre -1 y 1. Tiene la ventaja de que al estar centrada en 0 tiene gradientes de mayor tamaño pero termina sufriendo los mismos problemas en sus extremos.



Softmax. Generalización de la función sigmoid para resolver problemas de clasificación multiclase. Produce una probabilidad entre todas las salidas igual a 1(80% perro,  20% gato). 

RELU. La Unidad Rectificadora Lineal es una función cuyo uso se vuelve casi obligatorio en redes multicapas dejando el uso de otras funciones únicamente para la salida de la red.



Como se observa, esta función es parecida a la identidad para valores positivos de entrada, pero añade no-linealidad al devolver cero para valores negativos de entrada. Su derivada sólo puede tomar los valores constantes 1 o 0 y por tanto esta función no sufre de desvanecimiento ni explosión de gradiente. 
La activaciones a 0 mejoran la convergencia de la red debido a que fuerza a un conjunto más pequeño de neuronas a aprender de manera óptima, ya que en ellas recae la aportación del gradiente. 

El reducido número de activaciones producido por las salidas a 0 introduce velocidad y un menor uso de memoria en la red. También mejoran la convergencia forzando a las neuronas activadas a aprender de manera óptima, ya que en ellas recae la aportación del gradiente. Por tanto RELU simplifica el modelo y lo vuelve más ligero en comparación con sigmoid o tanh, donde la cantidad de activaciones es más densa y el cálculo del gradiente se vuelve más complejo.

Existen otras funciones parecidas como LRELU y PRELU que tratan de evitar la activaciones a 0 en caso de que se vuelva un problema para el aprendizaje de la red (RELU moribundo), pero no suelen resultar en mejoras significativas de los resultados[3].

Otra función que está obteniendo éxito es la unidad exponencial lineal o ELU que ha mostrado en alguno problemas ser mejor que la unidad rectificadora[4]

