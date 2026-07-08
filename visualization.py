"""
visualization
=============

Visualizaciones del grafo extraido de una imagen. Portado del bloque de dibujo
que estaba comentado en ``main.py``.
"""

import networkx as nx
import matplotlib.pyplot as plt


def plot_network_on_image(SA, xlim=None, ylim=None, transparent=False,
                          save_path=None, dpi=200, ax=None):
    """
    Dibuja el grafo de una ``SkeletonAnalysis`` sobre su imagen original, con los
    electrodos Vin (rojo) y Vout (azul) resaltados.

    Parametros
    ----------
    SA : SkeletonAnalysis
        Instancia ya procesada (con ``G``, ``node_coords``, ``electrode_pos``, ``img``).
    xlim, ylim : tuple opcional
        Limites para hacer zoom en una region.
    transparent : bool
        Si True, oculta la imagen de fondo y hace transparente la figura
        (util para exportar solo el grafo).
    save_path : str opcional
        Si se pasa, guarda la figura.
    dpi : int
        Resolucion al guardar.
    ax : matplotlib Axes opcional
        Eje donde dibujar; si no se pasa, se crea uno nuevo.
    """
    pos = {}
    for n in SA.G.nodes:
        if isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")):
            pos[n] = SA.electrode_pos[n]
        else:
            y, x = SA.node_coords[n]
            pos[n] = (x, y)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(SA.img, cmap="gray", alpha=0 if transparent else 1)

    nodos_vin = [n for n in SA.G.nodes if isinstance(n, str) and n.startswith("Vin_")]
    nodos_vout = [n for n in SA.G.nodes if isinstance(n, str) and n.startswith("Vout_")]
    nodos_red = [n for n in SA.G.nodes
                 if not (isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")))]

    nx.draw_networkx_edges(SA.G, pos, ax=ax, width=1.5, edge_color="green")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=nodos_red, node_size=25, node_color="C0")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=nodos_vin, node_size=150, node_color="red")
    nx.draw_networkx_nodes(SA.G, pos, ax=ax, nodelist=nodos_vout, node_size=150, node_color="blue")

    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if transparent:
        ax.figure.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return ax
