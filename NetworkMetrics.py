
import numpy as np
import networkx as nx
from networkx.algorithms import community

class NetworkMetrics:
    def __init__(self, G):
        self.G = G.copy()

    def get_avg_degree(self):
        """Calcula el grado promedio de la red."""
        degrees = [d for n, d in self.G.degree()]
        return np.mean(degrees)

    def get_density(self):
        """Densidad de la red (aristas existentes vs posibles)."""
        return nx.density(self.G)

    def get_avg_clustering(self):
        """Coeficiente de clustering promedio (CCoeff)."""
        return nx.average_clustering(self.G)

    def get_avg_path_length(self):
        """Longitud de camino promedio (PL). 
        Nota: Solo funciona si la red es conexa (todos conectados)."""
        if nx.is_connected(self.G):
            return nx.average_shortest_path_length(self.G)
        else:
            # Si hay islas, calculamos sobre la componente gigante
            largest_cc = max(nx.connected_components(self.G), key=len)
            subgraph = self.G.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph)

    def get_modularity(self):
        """
        Calcula la modularidad detectando comunidades automáticamente.
        Usa el algoritmo Greedy (rápido y estándar).
        """
        try:
            # Detectar comunidades
            communities = community.greedy_modularity_communities(self.G)
            # Calcular score de modularidad
            return community.modularity(self.G, communities)
        except:
            return np.nan

    def get_small_world_sigma(self):
        """
        Small-worldness (Sigma).
        Compara el clustering y path length con un grafo aleatorio equivalente.
        sigma > 1 indica small-world.
        ADVERTENCIA: Es lento en grafos grandes (>500 nodos).
        """
        if len(self.G) > 500: return "Slow/Skip" # Omitir si es muy grande
        if not nx.is_connected(self.G): return np.nan
        try:
            # niter=10 es bajo para velocidad, subir a 100 para precisión
            return nx.sigma(self.G, niter=10, nrand=5) 
        except:
            return np.nan

    def get_global_efficiency(self):
        """Eficiencia Global (inverso del path length). Ideal para redes desconectadas."""
        return nx.global_efficiency(self.G)

    def get_assortativity(self):
        """Si los nodos se conectan con otros de grado similar (Hubs con Hubs)."""
        return nx.degree_assortativity_coefficient(self.G)
    
    def run_metrics(self, verbose=False):
        """Ejecuta todos los cálculos y devuelve un diccionario con los resultados."""
        metrics = {
            "Average Degree": self.get_avg_degree(),
            "Density": self.get_density(),
            "Average Clustering Coefficient": self.get_avg_clustering(),
            "Average Path Length": self.get_avg_path_length(),
            "Modularity": self.get_modularity(),
            "Small-World Sigma": self.get_small_world_sigma(),
            "Global Efficiency": self.get_global_efficiency(),
            "Assortativity": self.get_assortativity()
        }
        if verbose:
            for key, value in metrics.items():
                print(f"{key}: {value}")
        return metrics