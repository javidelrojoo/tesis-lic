
import numpy as np
import networkx as nx
from networkx.algorithms import community

class NetworkMetrics:
    def __init__(self, G):
        self.G = G.copy()

    def get_avg_degree(self, return_var = False):
        """Calcula el grado promedio de la red."""
        degrees = [d for n, d in self.G.degree()]
        if return_var:
            return np.mean(degrees), np.var(degrees)
        
        return np.mean(degrees)
    
    def get_degree_hist(self):
        return nx.degree_histogram(self.G)

    def get_density(self):
        """Densidad de la red (aristas existentes vs posibles)."""
        return nx.density(self.G)

    def get_avg_clustering(self):
        """Coeficiente de clustering promedio (CCoeff)."""
        return nx.average_clustering(self.G)

    def get_communities(self, algorithm = 'greedy'):
        if algorithm == "greedy":
            communities_raw = community.greedy_modularity_communities(self.G)
            self.communities = {
                node: i
                for i, comm in enumerate(communities_raw)
                for node in comm
            }
            return self.communities

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
        Calcula la modularidad..
        """
        return community.modularity(self.G, self.communities)
        

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