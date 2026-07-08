"""
NetworkMetrics
==============

Metricas de calculo para las redes de nanohilos (y modelos de referencia) de la
tesis. Consolida en un solo lugar todo el procesamiento que estaba disperso en
``Codigos/03_analisis/graph_analysis.py``.

Diseno:
    - Las metricas viven como funciones libres (una sola fuente de verdad).
    - La clase ``NetworkMetrics`` envuelve un grafo y delega en esas funciones,
      cacheando las comunidades para no recalcularlas.

Convencion importante (path length / global efficiency):
    Se calculan con ``scipy.sparse.csgraph.shortest_path`` sobre la componente
    conexa mas grande y por muestreo de nodos fuente (metodo "sampled"), tal como
    en graph_analysis.py. Es la forma que se usa por defecto porque escala a las
    redes grandes de nanohilos. Para redes chicas el muestreo (k=1000) cae sobre
    todos los nodos y el resultado coincide con el exacto.

    Ademas se incluye la variante "electrodos" (source->target, tipico Vin<->Vout)
    y una version exacta con networkx/igraph para redes chicas.
"""

import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
from scipy.sparse.csgraph import shortest_path


# =============================================================================
#  Metricas basicas
# =============================================================================

def get_num_nodes(G):
    """Cantidad de nodos."""
    return G.number_of_nodes()


def get_num_edges(G):
    """Cantidad de aristas."""
    return G.number_of_edges()


def get_avg_degree(G, return_var=False):
    """Grado promedio de la red. Si ``return_var`` es True devuelve (media, varianza)."""
    degrees = [d for n, d in G.degree()]
    if return_var:
        return np.mean(degrees), np.var(degrees)
    return np.mean(degrees)


def get_degree_variance(G):
    """Varianza de la distribucion de grados."""
    degrees = [d for n, d in G.degree()]
    return np.var(degrees)


def get_degree_histogram(G):
    """Histograma de grados (lista donde el indice i son los nodos con grado i)."""
    return nx.degree_histogram(G)


def get_density(G):
    """Densidad de la red (aristas existentes vs posibles)."""
    return nx.density(G)


def get_avg_clustering(G):
    """Coeficiente de clustering promedio (CCoeff)."""
    return nx.average_clustering(G)


def get_assortativity(G):
    """Asortatividad de grado: si los nodos se conectan con otros de grado similar."""
    return nx.degree_assortativity_coefficient(G)


# =============================================================================
#  Comunidades y modularidad
# =============================================================================

def detect_communities(G, algorithm="greedy"):
    """
    Detecta comunidades y devuelve una particion (lista de conjuntos de nodos).

    algorithm: "greedy" (greedy modularity), "louvain" o "label_propagation".
    """
    if algorithm == "greedy":
        return list(community.greedy_modularity_communities(G))
    if algorithm == "louvain":
        return list(community.louvain_communities(G))
    if algorithm == "label_propagation":
        return list(community.label_propagation_communities(G))
    raise ValueError(
        f"algorithm debe ser 'greedy', 'louvain' o 'label_propagation', recibido: '{algorithm}'"
    )


def communities_as_dict(communities):
    """Convierte una particion (lista de conjuntos) en un dict {nodo: id_comunidad}."""
    return {nodo: i for i, comm in enumerate(communities) for nodo in comm}


def get_modularity(G, communities=None, algorithm="greedy"):
    """
    Modularidad de una particion.

    ``communities`` puede ser:
        - None: se detecta automaticamente con ``algorithm``.
        - una lista de conjuntos de nodos (particion).
        - un dict {nodo: id_comunidad}.
    """
    if communities is None:
        communities = detect_communities(G, algorithm)
    elif isinstance(communities, dict):
        grupos = {}
        for nodo, cid in communities.items():
            grupos.setdefault(cid, set()).add(nodo)
        communities = list(grupos.values())

    try:
        return community.modularity(G, communities)
    except Exception:
        return np.nan


# =============================================================================
#  Path length y global efficiency (metodo de graph_analysis.py)
#
#  Ambas metricas comparten la matriz de distancias, asi que se calculan juntas
#  en una sola pasada. metric = "path_length" | "global_efficiency" | "both".
# =============================================================================

def get_electrode_nodes(G):
    """Devuelve (Vin_list, Vout_list): nodos electrodo detectados por nombre."""
    Vin = [n for n in G.nodes() if "Vin" in str(n)]
    Vout = [n for n in G.nodes() if "Vout" in str(n)]
    return Vin, Vout


def path_length_efficiency_sampled(G, k=1000, seed=42, metric="both"):
    """
    Path length y/o global efficiency sobre la componente conexa mas grande,
    muestreando ``k`` nodos fuente (BFS multi-source con scipy).

    Escala a redes grandes. Con k >= N_componente equivale al calculo exacto.
    """
    random.seed(seed)

    components = list(nx.connected_components(G))
    if not components:
        return (float("inf"), float("inf")) if metric == "both" else float("inf")

    Gc = G.subgraph(max(components, key=len)).copy()
    nodes = list(Gc.nodes)

    if len(nodes) < 2:
        return (float("inf"), float("inf")) if metric == "both" else float("inf")

    sample = random.sample(nodes, min(k, len(nodes)))
    node_idx = {n: i for i, n in enumerate(nodes)}
    src_idx = [node_idx[s] for s in sample]

    A = nx.to_scipy_sparse_array(Gc)
    D = shortest_path(A, indices=src_idx, directed=False)

    mask = (D != np.inf) & (D != 0)
    if not mask.any():
        return (float("inf"), float("inf")) if metric == "both" else float("inf")

    # Path length: promedio de distancias finitas no nulas
    pl = D[mask].mean()

    # Global efficiency: promedio de las distancias inversas
    with np.errstate(divide="ignore"):
        inv_D = 1.0 / D
    inv_D[np.isinf(inv_D)] = 0
    N = len(nodes)
    ge = inv_D.sum() / (N * (N - 1))

    if metric == "path_length":
        return pl
    if metric == "global_efficiency":
        return ge
    if metric == "both":
        return pl, ge
    raise ValueError(
        f"metric debe ser 'path_length', 'global_efficiency' o 'both', recibido: '{metric}'"
    )


def path_length_efficiency_src2tgt(G, sources, targets, directed=False, metric="both"):
    """
    Path length y/o global efficiency entre dos conjuntos de nodos
    (tipicamente electrodos Vin -> Vout).
    """
    nodes = list(G.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    src_idx = [node_idx[s] for s in sources]
    tgt_idx = [node_idx[t] for t in targets]

    A = nx.to_scipy_sparse_array(G)
    D = shortest_path(A, indices=src_idx, directed=directed)

    sub = D[np.ix_(range(len(src_idx)), tgt_idx)]
    lengths = sub.flatten()
    lengths = lengths[lengths != np.inf]

    if not len(lengths):
        return (float("inf"), float("inf")) if metric == "both" else float("inf")

    # Path length
    pl = lengths.mean()

    # Global efficiency
    with np.errstate(divide="ignore"):
        inv_sub = 1.0 / sub
    inv_sub[np.isinf(inv_sub)] = 0
    N = len(sources) * len(targets)
    ge = inv_sub.sum() / N if N else float("inf")

    if metric == "path_length":
        return pl
    if metric == "global_efficiency":
        return ge
    if metric == "both":
        return pl, ge
    raise ValueError(
        f"metric debe ser 'path_length', 'global_efficiency' o 'both', recibido: '{metric}'"
    )


def get_avg_path_length(G, k=1000, seed=42):
    """Longitud de camino promedio (L), metodo sampleado sobre componente gigante."""
    return path_length_efficiency_sampled(G, k=k, seed=seed, metric="path_length")


def get_global_efficiency(G, k=1000, seed=42):
    """Eficiencia global (E), metodo sampleado sobre componente gigante."""
    return path_length_efficiency_sampled(G, k=k, seed=seed, metric="global_efficiency")


def get_avg_path_length_electrodes(G, directed=False):
    """Longitud de camino promedio entre electrodos (Vin <-> Vout). inf si no hay electrodos."""
    Vin, Vout = get_electrode_nodes(G)
    if not (Vin and Vout):
        return float("inf")
    return path_length_efficiency_src2tgt(G, Vin, Vout, directed=directed, metric="path_length")


def get_global_efficiency_electrodes(G, directed=False):
    """Eficiencia global entre electrodos (Vin <-> Vout). inf si no hay electrodos."""
    Vin, Vout = get_electrode_nodes(G)
    if not (Vin and Vout):
        return float("inf")
    return path_length_efficiency_src2tgt(G, Vin, Vout, directed=directed, metric="global_efficiency")


def get_avg_path_length_exact(G):
    """
    Longitud de camino promedio EXACTA con networkx.
    Si la red no es conexa, se calcula sobre la componente gigante.
    Solo recomendable para redes chicas (es O(N^2) o peor).
    """
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    componentes = sorted(nx.connected_components(G), key=len, reverse=True)
    componente_gigante = G.subgraph(componentes[0])
    return nx.average_shortest_path_length(componente_gigante)


def get_avg_path_length_igraph(G):
    """
    Longitud de camino promedio EXACTA con igraph (mas rapido que networkx).
    Se calcula sobre la componente gigante si la red no es conexa.
    Requiere el paquete ``igraph`` (import diferido para no forzar la dependencia).
    """
    import igraph as ig

    G_ig = ig.Graph.from_networkx(G)
    if G_ig.is_connected():
        return G_ig.average_path_length()
    return G_ig.connected_components().giant().average_path_length()


def get_network_resistance(G, R_default=1.0):
    """
    Resistencia de dos terminales entre electrodos (Vin -> Vout), resolviendo la
    red como un circuito resistivo por el metodo del potencial de nodos
    (Laplaciana pesada por conductancias).

    Se unen todos los Vin a un supernodo SOURCE y todos los Vout a un SINK, se
    inyecta 1 A y se mide la tension: con I = 1, V = R. El peso de arista
    ``weight`` se interpreta como resistencia (``R_default`` si falta).

    Devuelve inf si la red esta abierta entre electrodos. Lanza ValueError si no
    hay electrodos.
    """
    Vin, Vout = get_electrode_nodes(G)
    if not Vin or not Vout:
        raise ValueError("No se encontraron nodos de entrada (Vin) o salida (Vout).")

    G_work = G.copy()
    for n in Vin:
        G_work.add_edge("SOURCE", n, weight=1e-12)   # resistencia despreciable
    for n in Vout:
        G_work.add_edge("SINK", n, weight=1e-12)

    # Si no hay camino entre electrodos la resistencia es infinita (red abierta).
    if not nx.has_path(G_work, "SOURCE", "SINK"):
        return float("inf")

    for u, v, data in G_work.edges(data=True):
        r_val = data.get("weight", R_default)
        data["conductance"] = 1.0 / r_val if r_val > 0 else 1e12

    nodos = list(G_work.nodes())
    idx_source = nodos.index("SOURCE")
    idx_sink = nodos.index("SINK")

    Lap = nx.laplacian_matrix(G_work, weight="conductance").toarray()
    # Fijamos el SINK a 0 V eliminando su fila/columna
    L_red = np.delete(np.delete(Lap, idx_sink, axis=0), idx_sink, axis=1)

    I_red = np.zeros(len(nodos) - 1)
    new_idx_source = idx_source if idx_source < idx_sink else idx_source - 1
    I_red[new_idx_source] = 1.0  # inyectamos 1 A en SOURCE

    try:
        V_red = np.linalg.solve(L_red, I_red)
        return float(V_red[new_idx_source])  # V = R cuando I = 1
    except np.linalg.LinAlgError:
        return float("inf")  # red abierta


# =============================================================================
#  Graficos
#
#  Todas las funciones aceptan un ``ax`` opcional (para componer figuras) y lo
#  devuelven. Si no se pasa, crean su propia figura.
# =============================================================================

def plot_degree_distribution(G, ax=None, color="tab:blue", alpha=1.0, title=None, logy=True):
    """Distribucion de grados (histograma en escala log por defecto)."""
    dh = get_degree_histogram(G)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.bar(np.arange(len(dh)), dh, color=color, edgecolor="black", alpha=alpha)
    ax.set_xlabel("Grado")
    ax.set_ylabel("Cuentas")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return ax


def plot_adjacency_matrix(G, communities=None, algorithm="greedy", ax=None,
                          color="tab:blue", markersize=0.1, title=None):
    """
    Matriz de adyacencia (spy) con los nodos reordenados por comunidad, de modo
    que la estructura modular aparece como bloques en la diagonal.

    ``communities`` puede ser None (se detecta), una particion (lista de conjuntos)
    o un dict {nodo: id_comunidad}.
    """
    if communities is None:
        communities = detect_communities(G, algorithm)

    if isinstance(communities, dict):
        order = sorted(G.nodes(), key=lambda n: communities[n])
    else:
        order = []
        for comm in sorted(communities, key=len, reverse=True):
            order.extend(list(comm))

    A = nx.to_scipy_sparse_array(G, nodelist=order)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.spy(A, markersize=markersize, color=color)
    if title:
        ax.set_title(title)
    return ax


def plot_community_sizes(communities, ax=None, color="tab:blue", alpha=1.0,
                         bins=50, title=None, logy=True):
    """Distribucion de tamanos de comunidad. ``communities`` = particion o dict."""
    if isinstance(communities, dict):
        from collections import Counter
        sizes = list(Counter(communities.values()).values())
    else:
        sizes = [len(c) for c in communities]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.hist(sizes, bins=bins, color=color, edgecolor="black", alpha=alpha)
    ax.set_xlabel("Tamano de comunidad")
    ax.set_ylabel("Cuentas")
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    return ax


def plot_path_length_distribution(G, k=1000, seed=42, ax=None, color="tab:blue", title=None):
    """
    Distribucion de longitudes de camino (metodo sampleado sobre componente
    gigante), con media, mediana y desvio marcados. Devuelve (ax, stats).
    """
    random.seed(seed)
    components = list(nx.connected_components(G))
    if not components:
        return ax, None

    Gc = G.subgraph(max(components, key=len)).copy()
    nodes = list(Gc.nodes)
    if len(nodes) < 2:
        return ax, None

    sample = random.sample(nodes, min(k, len(nodes)))
    node_idx = {n: i for i, n in enumerate(nodes)}
    src_idx = [node_idx[s] for s in sample]

    A = nx.to_scipy_sparse_array(Gc)
    D = shortest_path(A, indices=src_idx, directed=False)

    mask = (D != np.inf) & (D != 0)
    if not mask.any():
        return ax, None

    lengths = D[mask].astype(int)
    mean = float(lengths.mean())
    median = float(np.median(lengths))
    std = float(lengths.std())

    if ax is None:
        _, ax = plt.subplots()
    ax.hist(lengths, bins=range(lengths.min(), lengths.max() + 2),
            edgecolor="white", density=True, color=color)
    ax.axvline(mean, color="red", linestyle="--", label=f"Media: {mean:.2f}")
    ax.axvline(median, color="orange", linestyle="--", label=f"Mediana: {median:.2f}")
    ax.axvspan(mean - std, mean + std, alpha=0.15, color="red", label=f"Std: {std:.2f}")
    ax.set_xlabel("Path length")
    ax.set_ylabel("Frecuencia relativa")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax, {"mean": mean, "median": median, "std": std}


# =============================================================================
#  Tabla comparativa
# =============================================================================

# Lista canonica (Nombre, Funcion) usada para tablas comparativas.
METRICAS = [
    ("Avg Degree (Grado)", get_avg_degree),
    ("Density (Densidad)", get_density),
    ("Avg Clustering (CCoeff)", get_avg_clustering),
    ("Avg Path Length (L)", get_avg_path_length),
    ("Global Efficiency (E)", get_global_efficiency),
    ("Modularity", get_modularity),
    ("Assortativity", get_assortativity),
]


def compare_graphs(grafos, metricas=METRICAS):
    """
    Imprime una tabla comparando las metricas de varios grafos.
    ``grafos`` es un dict {nombre: grafo}. Devuelve el dict de resultados.
    """
    nombres = list(grafos.keys())

    encabezado = f"{'Metrica':<25} | " + " | ".join(f"{n:<12}" for n in nombres)
    print(encabezado)
    print("-" * len(encabezado))

    resultados = {}
    for nombre_metrica, funcion in metricas:
        fila = []
        resultados[nombre_metrica] = {}
        for nombre_grafo, G in grafos.items():
            val = funcion(G)
            resultados[nombre_metrica][nombre_grafo] = val
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            fila.append(f"{val_str:<12}")
        print(f"{nombre_metrica:<25} | " + " | ".join(fila))

    return resultados


# =============================================================================
#  Clase envoltorio
# =============================================================================

class NetworkMetrics:
    """Envuelve un grafo y expone las metricas de calculo, cacheando comunidades."""

    def __init__(self, G):
        self.G = G.copy()
        self._communities = None

    # --- Basicas ---
    def get_num_nodes(self):
        return get_num_nodes(self.G)

    def get_num_edges(self):
        return get_num_edges(self.G)

    def get_avg_degree(self, return_var=False):
        return get_avg_degree(self.G, return_var=return_var)

    def get_degree_variance(self):
        return get_degree_variance(self.G)

    def get_degree_histogram(self):
        return get_degree_histogram(self.G)

    def get_density(self):
        return get_density(self.G)

    def get_avg_clustering(self):
        return get_avg_clustering(self.G)

    def get_assortativity(self):
        return get_assortativity(self.G)

    # --- Comunidades ---
    def detect_communities(self, algorithm="greedy"):
        self._communities = detect_communities(self.G, algorithm)
        return self._communities

    def get_communities_dict(self, algorithm="greedy"):
        if self._communities is None:
            self.detect_communities(algorithm)
        return communities_as_dict(self._communities)

    def get_modularity(self, algorithm="greedy"):
        if self._communities is None:
            self.detect_communities(algorithm)
        return get_modularity(self.G, self._communities)

    # --- Path length / efficiency ---
    def get_electrode_nodes(self):
        return get_electrode_nodes(self.G)

    def get_avg_path_length(self, k=1000, seed=42):
        return get_avg_path_length(self.G, k=k, seed=seed)

    def get_global_efficiency(self, k=1000, seed=42):
        return get_global_efficiency(self.G, k=k, seed=seed)

    def get_avg_path_length_electrodes(self, directed=False):
        return get_avg_path_length_electrodes(self.G, directed=directed)

    def get_global_efficiency_electrodes(self, directed=False):
        return get_global_efficiency_electrodes(self.G, directed=directed)

    def get_avg_path_length_exact(self):
        return get_avg_path_length_exact(self.G)

    def get_avg_path_length_igraph(self):
        return get_avg_path_length_igraph(self.G)

    def get_network_resistance(self, R_default=1.0):
        return get_network_resistance(self.G, R_default=R_default)

    # --- Graficos ---
    def plot_degree_distribution(self, ax=None, color="tab:blue", alpha=1.0, title=None, logy=True):
        return plot_degree_distribution(self.G, ax=ax, color=color, alpha=alpha, title=title, logy=logy)

    def plot_adjacency_matrix(self, algorithm="greedy", ax=None, color="tab:blue",
                              markersize=0.1, title=None):
        if self._communities is None:
            self.detect_communities(algorithm)
        return plot_adjacency_matrix(self.G, communities=self._communities, ax=ax,
                                     color=color, markersize=markersize, title=title)

    def plot_community_sizes(self, algorithm="greedy", ax=None, color="tab:blue",
                             alpha=1.0, bins=50, title=None, logy=True):
        if self._communities is None:
            self.detect_communities(algorithm)
        return plot_community_sizes(self._communities, ax=ax, color=color, alpha=alpha,
                                    bins=bins, title=title, logy=logy)

    def plot_path_length_distribution(self, k=1000, seed=42, ax=None, color="tab:blue", title=None):
        return plot_path_length_distribution(self.G, k=k, seed=seed, ax=ax, color=color, title=title)

    # --- Resumen ---
    def run_metrics(self, verbose=False):
        """Ejecuta los calculos principales y devuelve un diccionario con los resultados."""
        avg_deg, var_deg = self.get_avg_degree(return_var=True)
        metrics = {
            "Num Nodes": self.get_num_nodes(),
            "Num Edges": self.get_num_edges(),
            "Average Degree": avg_deg,
            "Degree Variance": var_deg,
            "Density": self.get_density(),
            "Average Clustering Coefficient": self.get_avg_clustering(),
            "Average Path Length": self.get_avg_path_length(),
            "Global Efficiency": self.get_global_efficiency(),
            "Modularity": self.get_modularity(),
            "Assortativity": self.get_assortativity(),
        }

        Vin, Vout = self.get_electrode_nodes()
        if Vin and Vout:
            metrics["Average Path Length (Vin<->Vout)"] = self.get_avg_path_length_electrodes()
            metrics["Global Efficiency (Vin<->Vout)"] = self.get_global_efficiency_electrodes()
            try:
                metrics["Network Resistance (Vin->Vout)"] = self.get_network_resistance()
            except Exception:
                pass

        if verbose:
            for key, value in metrics.items():
                print(f"{key}: {value}")
        return metrics
