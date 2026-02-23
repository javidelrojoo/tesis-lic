import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

path = 'muestras/M3/M30007.graphml'

G = nx.Graph(nx.read_graphml(path))

ohmic_ratio = 0.5

for node in G.nodes():
    if np.random.rand() < ohmic_ratio:
        neighbors = list(G.neighbors(node))