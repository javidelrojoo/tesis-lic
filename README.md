# Obtención de grafos de conexión de autoensamblados neuromórficos a partir de sus fotomicrografías
#### Javier Tau Anzoátegui. 
##### Lugar de trabajo: LINE, ECyT-UNSAM. Directores: Cynthia Quinteros y Oscar Filevich.
## Resumen del proyecto
En el marco de una línea de investigación orientada al aprovechamiento de sistemas autoensamblados para el
procesamiento de señales, este proyecto se propone caracterizar la interconexión presente en ciertos
autoensamblados y el impacto en sus propiedades de interés.
Se prevé la adquisición de imágenes, previa optimización de las condiciones empleadas para ello. Las imágenes
resultantes serán analizadas empleando estrategias similares a las de tejidos neuronales. Se espera que la
modalidad de adquisición permita cuantificar el grado de interconexión de los autoensamblados mediante un
postprocesamiento computacional a desarrollar. Contando con una imagen como insumo, se espera que dicho
procesamiento permita identificar conexiones e intersecciones que se traduzcan en la generación de un grafo
(objeto matemático compuesto por nodos y aristas) representativo del autoensamblado experimental.
La obtención de dicho grafo o diagrama de conexión permitirá evaluar la red en términos biológicos y
electrónicos. Por un lado, la morfología y el grado de interconexión pueden compararse con la de sistemas
neuromórficos biológicos. Por otro, la obtención del grafo permite articular con un proyecto preexistente de
simulación del comportamiento eléctrico de ensambles memristivos y así simular el flujo de información en un sistema experimental.

# Obtaining connection graphs of neuromorphic self-assemblies from their photomicrographs
Within the framework of a research line focused on harnessing self-assembled systems for signal processing, this project aims to characterize the interconnection present in certain self-assemblies and its impact on their properties of interest.  
Image acquisition is planned, preceded by the optimization of the experimental conditions employed. The resulting images will be analyzed using strategies analogous to those applied to neuronal tissues. It is expected that the chosen acquisition modality will enable quantification of the degree of interconnection in the self-assemblies through computational post-processing to be developed. Using an image as input, such processing is expected to identify connections and intersections, leading to the generation of a graph (a mathematical object composed of nodes and edges) representative of the experimental self-assembly.  
The construction of such a graph or connection diagram will allow the network to be assessed in both biological and electronic terms. On one hand, the morphology and degree of interconnection can be compared to those of biological neuromorphic systems. On the other, the graph will provide a link to an existing project on the simulation of the electrical behavior of memristive assemblies, thus enabling the simulation of information flow in an experimental system.

# Tutorial de Uso

### Paso 1: Instalación de dependencias
Para ejecutar este código, asegurate de tener instaladas las siguientes librerías de Python. Podés instalarlas ejecutando:

```bash
pip install opencv-python numpy scikit-image skan networkx matplotlib
```

### Paso 2: Inicialización

En tu archivo principal (por ejemplo, `main.py`), instanciá la clase `SkeletonAnalysis` indicando la ruta de la imagen que querés procesar.

> **Nota:** La imagen de entrada ya debe estar binarizada y preferentemente en formato `.tif`.

```python
from SkeletonAnalysis import SkeletonAnalysis 

ruta_imagen = "ruta/a/tu/imagen_binarizada.tif"
SA = SkeletonAnalysis(ruta_imagen)

```

### Paso 3: Procesamiento y extracción del grafo

Una vez creado el objeto, podés ejecutar el pipeline completo de manera automática con un solo comando, o bien ejecutar cada método paso a paso si necesitás mayor control sobre el procesamiento de la red.

**Opción A: Análisis completo (Recomendado)**

```python
SA.complete_analysis(
    electrodes_path=("ruta/mascara_Vin.tif", "ruta/mascara_Vout.tif"),
    save_graph_path="grafo_salida.graphml",
    plot=True,
    verbose=True
)

```

**Opción B: Paso a paso**

```python
analisis.img_cleaning()
analisis.skeletonize()
analisis.get_network()
analisis.clean_network()
analisis.plot_graph()

```

---

## Referencia de la Clase `SkeletonAnalysis`

A continuación se detallan los métodos disponibles dentro de la clase para el procesamiento morfológico y la extracción topológica de la red:

* **`__init__(self, img_path)`**: Constructor de la clase. Carga la imagen binarizada en escala de grises y establece la bandera de electrodos en `False`.
* **`load_electrodes(self, path1, path2)`**: Carga las máscaras correspondientes a los electrodos de entrada y salida. Esto es fundamental si se busca modelar el flujo a través de la red de nanohilos.
* **`img_cleaning(self)`**: Realiza una limpieza morfológica de la imagen. Elimina pequeños agujeros (ruido) y aplica una operación de apertura (opening) con un disco de radio 1 para suavizar las estructuras antes de esqueletizar.
* **`skeletonize(self)`**: Reduce las estructuras de la imagen binarizada a un ancho de un solo píxel (esqueleto). Si se cargaron electrodos, asegura la conectividad dilatando las áreas de contacto para unir el esqueleto principal con las zonas de estímulo.
* **`get_network(self)`**: Utiliza la librería `skan` para transformar el esqueleto en un grafo de NetworkX (`self.G`). Extrae las coordenadas espaciales de los nodos. Si hay electrodos, genera automáticamente los nodos de entrada y de salida (`Vin` y `Vout`).
* **`clean_network(self)`**: Optimiza la topología del grafo extraído de forma iterativa:
    * Poda los nodos de grado 1 (ramas muertas o "dead ends").
    * Colapsa los nodos de grado 2, simplificando caminos continuos en una sola arista entre intersecciones.
    * Elimina los bucles o auto-conexiones (self-loops).


* **`convert_to_line_graph(self)`**: Transforma el grafo espacial (`self.G`) en su grafo de líneas correspondiente (`self.L`).
* **`save_graph(self, path, save_line_graph=True)`**: Exporta la topología obtenida en formato `.graphml`. Por defecto guarda el grafo de líneas, pero puede configurarse para guardar el grafo espacial original.
* **`plot_graph(self, save_path=None)`**: Genera una visualización del grafo superpuesto a la micrografía original utilizando `matplotlib`. Calcula los momentos de las imágenes de los electrodos para posicionar correctamente los nodos `Vin` y `Vout`.
* **`complete_analysis(self, electrodes_path=None, save_graph_path=None, plot=False, save_plot_path=None, verbose=False)`**: Función integradora que ejecuta todo el pipeline de análisis en orden (limpieza, esqueletización, extracción, poda y conversión), con opciones para guardar los resultados y mostrar el progreso en consola.

