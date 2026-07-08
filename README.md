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

El procesamiento está organizado en módulos independientes que siguen el flujo
imagen → grafo → análisis:

| Módulo | Rol |
| --- | --- |
| `preprocessing.py` | Binarización de las fotomicrografías previa a la extracción. |
| `SkeletonAnalysis.py` | Esqueletización de la imagen y extracción del grafo (`.graphml`). |
| `main.py` | Punto de entrada: corre el pipeline de extracción sobre una muestra. |
| `NetworkMetrics.py` | Métricas de la red (grado, clustering, comunidades, path length, eficiencia, resistencia) y sus gráficos. |
| `edge_removal.py` | Experimentos de robustez por eliminación aleatoria de aristas y ajustes. |
| `graph_transforms.py` | Transformaciones / coarse-graining del grafo (fusión de nodos y de aristas). |
| `visualization.py` | Dibujo del grafo superpuesto a la imagen. |

Flujo típico: `preprocessing` → `SkeletonAnalysis` (vía `main.py`) → `NetworkMetrics` / `edge_removal` / `graph_transforms` / `visualization`.

### Paso 1: Instalación de dependencias
Para ejecutar este código, asegurate de tener instaladas las siguientes librerías de Python. Podés instalarlas ejecutando:

```bash
pip install opencv-python numpy scipy scikit-image skan networkx matplotlib tqdm
```

> **Opcional:** para el cálculo exacto de path length con `igraph`
> (`NetworkMetrics.get_avg_path_length_igraph`) instalá además `python-igraph`.

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

---

## Preprocesamiento (`preprocessing.py`)

Prepara las fotomicrografías antes de pasarlas a `SkeletonAnalysis` (que espera una imagen ya binarizada).

```python
import preprocessing as pre

img = pre.load_gray("muestras/M4/M4.tif")
mask = pre.adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55)
overlay = pre.overlay_mask(img, mask)   # para revisar la binarización sobre la original
```

* **`load_gray(path)`**: Carga una imagen en escala de grises.
* **`adaptive_binarize(img, block_size=151, C=15, intensity_thresh=55)`**: Binarización adaptativa gaussiana (con *padding* replicado) que además fuerza a fondo los píxeles por debajo de `intensity_thresh`. Devuelve la máscara binaria (`0`/`255`).
* **`overlay_mask(base_gray, binary_mask, color_rgb=(220, 20, 60))`**: Superpone en color la máscara sobre la imagen en gris para inspección visual.

---

## Análisis de la red (`NetworkMetrics.py`)

Concentra todas las métricas de cálculo de la red. Cada métrica existe como función libre (`NetworkMetrics.get_...(G)`) y como método de la clase `NetworkMetrics`, que además cachea las comunidades detectadas.

```python
import networkx as nx
from NetworkMetrics import NetworkMetrics, compare_graphs

G = nx.read_graphml("muestras/M4/M4_effective.graphml")
m = NetworkMetrics(G)

m.run_metrics(verbose=True)     # resumen de todas las métricas
m.get_avg_path_length()         # path length (sampleado, escala a redes grandes)
m.get_network_resistance()      # resistencia Vin -> Vout como circuito resistivo
m.plot_degree_distribution()    # gráficos
m.plot_adjacency_matrix()

# comparar varias muestras en una tabla
compare_graphs({"M4": G})
```

**Métricas básicas:** `get_num_nodes`, `get_num_edges`, `get_avg_degree(return_var=False)`, `get_degree_variance`, `get_degree_histogram`, `get_density`, `get_avg_clustering`, `get_assortativity`.

**Comunidades:** `detect_communities(algorithm="greedy"|"louvain"|"label_propagation")`, `get_communities_dict()`, `get_modularity()`.

**Path length y eficiencia** (con `scipy.shortest_path` sobre la componente gigante, por muestreo de `k` nodos fuente; escala a redes grandes y con `k ≥ N` coincide con el cálculo exacto):

* `get_avg_path_length(k=1000, seed=42)` y `get_global_efficiency(k=1000, seed=42)`.
* Variante entre electrodos: `get_avg_path_length_electrodes()`, `get_global_efficiency_electrodes()`.
* Cálculo exacto para redes chicas: `get_avg_path_length_exact()` (networkx) y `get_avg_path_length_igraph()` (igraph, más rápido).

**Resistencia:** `get_network_resistance(R_default=1.0)` resuelve la red como circuito resistivo (Laplaciana pesada) y devuelve la resistencia de dos terminales Vin → Vout (`inf` si la red está abierta). Usa el atributo de arista `weight` como resistencia.

**Gráficos:** `plot_degree_distribution`, `plot_adjacency_matrix` (reordenada por comunidad), `plot_community_sizes`, `plot_path_length_distribution`. Todos aceptan un `ax` opcional y lo devuelven.

**Utilidades a nivel módulo:** `compare_graphs(grafos)` imprime una tabla comparativa; `METRICAS` es la lista canónica `(nombre, función)` que usa esa tabla.

---

## Experimentos de robustez (`edge_removal.py`)

Estudia cómo cambian las métricas al eliminar aristas al azar con probabilidad `p`. Las métricas se calculan reusando `NetworkMetrics`.

```python
import numpy as np
import edge_removal as er

grafos = {"M4": G}
ps = np.linspace(0.2, 0.5, 30)

res = er.edge_removal_dependency(grafos, ps=ps)      # una corrida
# res = er.edge_removal_variability(grafos, ps=ps, n_runs=20)  # con media ± desvío

er.plot_path_length_vs_p(res, ps=ps)
er.save_results(res, "edge_removal_M4.pkl")
```

* **`edge_removal_dependency(grafos, ps, k=1000, seed=...)`**: una corrida por cada `p`; devuelve un dict con `ccoeffs`, `paths`, `efficiencies` (y sus versiones de electrodos).
* **`edge_removal_variability(grafos, ps, n_runs=20, k=1000, seed=...)`**: repite `n_runs` veces cada `p`; cada métrica queda como array `(len(ps), n_runs)` para calcular media y desvío.
* **`plot_clustering_vs_p` / `plot_path_length_vs_p` / `plot_efficiency_vs_p`**: grafican vs `p`; funcionan con resultados de una corrida (1D) o de variabilidad (2D, con banda de ±1σ) y superponen la curva de electrodos si existe.
* **`save_results(res, path)` / `load_results(path)`**: persistencia con `pickle`.
* **`fit_path_length_powerlaw(ps, L)`** y **`fit_clustering_decay(ps, ccoeffs)`**: ajustes de curvas (`L = A·(x0 − p)^(−β)` y `C = C0·(1 − p)^α`).

---

## Transformaciones del grafo (`graph_transforms.py`)

Coarse-graining de la red. Ambas funciones operan sobre una copia y aceptan `seed` para reproducibilidad.

* **`merge_nodes(G, merge_prob=0.01, seed=None, verbose=False)`**: fusiona nodos contrayendo aristas al azar. Devuelve `(G_nuevo, n_fusiones)`.
* **`merge_edges_ohmic(G, ohmic_ratio=0.1, seed=None)`**: elimina nodos "óhmicos" al azar y conecta entre sí a sus vecinos (los reemplaza por un clique).

---

## Visualización (`visualization.py`)

* **`plot_network_on_image(SA, xlim=None, ylim=None, transparent=False, save_path=None, dpi=200, ax=None)`**: dibuja el grafo de una instancia `SkeletonAnalysis` sobre su imagen, con los electrodos Vin (rojo) y Vout (azul) resaltados. `xlim`/`ylim` permiten hacer zoom y `transparent=True` exporta solo el grafo con fondo transparente.

