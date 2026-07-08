"""
graph_transforms
================

Transformaciones / coarse-graining de la red de nanohilos. Portado desde las
celdas 'Node Merger' y 'Edge Merger' de ``temp.py``.

Todas las funciones operan sobre una copia del grafo (no lo modifican in-place)
y aceptan ``seed`` para reproducibilidad.
"""

import itertools

import numpy as np
import networkx as nx


def merge_nodes(G, merge_prob=0.01, seed=None, verbose=False):
    """
    Fusion aleatoria de nodos por contraccion de aristas.

    Recorre las aristas en orden aleatorio y, con probabilidad ``merge_prob``,
    contrae la arista (fusiona sus dos nodos en uno).

    Devuelve (G_nuevo, n_fusiones).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph(G).copy()

    edges = list(G.edges())
    rng.shuffle(edges)

    n0 = G.number_of_nodes()
    count = 0
    for u, v in edges:
        if G.has_node(u) and G.has_node(v):
            if rng.random() < merge_prob:
                nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
                count += 1

    if verbose:
        print(f"{n0} nodos iniciales -> {G.number_of_nodes()} restantes "
              f"({count} fusiones).")
    return G, count


def merge_edges_ohmic(G, ohmic_ratio=0.1, seed=None):
    """
    Eliminacion de nodos 'ohmicos' al azar.

    Con probabilidad ``ohmic_ratio`` por nodo, elimina el nodo y conecta entre si
    a todos sus vecinos (los reemplaza por un clique), modelando un contacto
    ohmico que fusiona sus conexiones.

    Devuelve el grafo transformado.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph(G).copy()

    for node in list(G.nodes()):
        if rng.random() < ohmic_ratio:
            neighbors = list(G.neighbors(node))
            G.add_edges_from(itertools.combinations(neighbors, 2))
            G.remove_node(node)

    return G
