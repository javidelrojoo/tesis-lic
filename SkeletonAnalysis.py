import cv2
import numpy as np
from skimage.morphology import remove_small_holes, opening, disk, skeletonize, dilation, remove_small_objects
from scipy.ndimage import binary_fill_holes
from skan import Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx
import matplotlib.pyplot as plt

class SkeletonAnalysis:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.has_electrodes = False
    
    def load_electrodes(self, path1, path2):
        self.electrode_mask1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        self.electrode_mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        self.electrode_mask1 = binary_fill_holes(self.electrode_mask1)
        self.electrode_mask2 = binary_fill_holes(self.electrode_mask2)

        self.electrode_mask1 = remove_small_objects(self.electrode_mask1, max_size=500)
        self.electrode_mask2 = remove_small_objects(self.electrode_mask2, max_size=500)

        self.has_electrodes = True
    
    def img_cleaning(self):
        self.img = remove_small_holes(self.img.astype(bool), max_size=2)
        self.img = opening(self.img, footprint=disk(1))
    
    def skeletonize(self):
        self.skeleton_img = skeletonize(self.img)
        if self.has_electrodes:
            # Solo eliminás los electrodos, sin agregar contact_points al skeleton
            self.skeleton_img = self.skeleton_img & ~self.electrode_mask1 & ~self.electrode_mask2

            # Guardás las máscaras dilatadas para usarlas en get_network
            self.electrode_dil1 = dilation(self.electrode_mask1, disk(5))
            self.electrode_dil2 = dilation(self.electrode_mask2, disk(5))
    
    def get_network(self):
        skeleton = Skeleton(self.skeleton_img)
        self.G = nx.Graph(skeleton_to_nx(skeleton))
        self.node_coords = np.array(skeleton.coordinates)

        if self.has_electrodes:
            electrode_nodes1 = []
            electrode_nodes2 = []

            for node_id, (y, x) in enumerate(self.node_coords):
                yi, xi = int(y), int(x)
                if self.electrode_dil1[yi, xi] and not self.electrode_mask1[yi, xi] and not self.electrode_mask2[yi, xi]:
                    electrode_nodes1.append(node_id)
                if self.electrode_dil2[yi, xi] and not self.electrode_mask2[yi, xi] and not self.electrode_mask1[yi, xi]:
                    electrode_nodes2.append(node_id)

            self.electrode_pos = {}

            for i, n in enumerate(electrode_nodes1):
                node_name = f"Vin_{i}"
                y, x = self.node_coords[n]
                self.G.add_node(node_name)
                self.G.add_edge(node_name, n)
                self.electrode_pos[node_name] = (x, y)

            for i, n in enumerate(electrode_nodes2):
                node_name = f"Vout_{i}"
                y, x = self.node_coords[n]
                self.G.add_node(node_name)
                self.G.add_edge(node_name, n)
                self.electrode_pos[node_name] = (x, y)
        
        self.G.remove_edges_from(list(nx.selfloop_edges(self.G)))    
        
    def clean_network(self):
        electrode_nodes = {n for n in self.G.nodes if isinstance(n, str) and 
                        (n.startswith("Vin_") or n.startswith("Vout_"))}
        changed = True
        while changed:
            changed = False

            isolated = [n for n, d in self.G.degree() if d == 0]
            if isolated:
                self.G.remove_nodes_from(isolated)
                changed = True

            deg1_nodes = [n for n, d in self.G.degree() if d == 1 and n not in electrode_nodes]
            if deg1_nodes:
                self.G.remove_nodes_from(deg1_nodes)
                changed = True

            deg2_nodes = [n for n, d in self.G.degree() if d == 2 and n not in electrode_nodes]
            for n in deg2_nodes:
                neighbors = list(self.G.neighbors(n))
                if len(neighbors) == 2:
                    u, v = neighbors
                    self.G.add_edge(u, v)
                self.G.remove_node(n)
                changed = True

            

            edges_to_remove = [(u, v) for u, v in self.G.edges()
                            if isinstance(u, str) and isinstance(v, str)]
            if edges_to_remove:
                self.G.remove_edges_from(edges_to_remove)
                changed = True
                
    
    def convert_to_line_graph(self):
        self.L = nx.line_graph(self.G)
        self.L = nx.Graph(self.L)
    
    def save_graph(self, path, save_line_graph=True):
        if save_line_graph:
            nx.write_graphml(self.L, path)
        else:
            nx.write_graphml(self.G, path)
    
    def plot_graph(self, save_path=None):
        pos = {}

        for n in self.G.nodes:
            if isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")):
                pos[n] = self.electrode_pos[n]
            else:
                y, x = self.node_coords[n]
                pos[n] = (x, y)

        plt.figure(figsize=(8, 8))
        plt.imshow(self.img, cmap='gray')

        nodos_vin    = [n for n in self.G.nodes if isinstance(n, str) and n.startswith("Vin_")]
        nodos_vout   = [n for n in self.G.nodes if isinstance(n, str) and n.startswith("Vout_")]
        nodos_red    = [n for n in self.G.nodes if not (isinstance(n, str) and (n.startswith("Vin_") or n.startswith("Vout_")))]

        nx.draw_networkx_edges(self.G, pos, width=1.5, edge_color='green', node_size=25)
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodos_red,  node_size=25)
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodos_vin,  node_size=150, node_color='red')
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodos_vout, node_size=150, node_color='blue')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.show()
    
    def complete_analysis(self, clean_network=True, electrodes_path=None, save_graph_path=None, plot=False, save_plot_path=None, verbose=False):
        if verbose:
            print("Iniciando análisis completo...")
        
        self.img_cleaning()
        if verbose:
            print("Imagen limpiada.")
        
        if electrodes_path:
            self.load_electrodes(*electrodes_path)
            if verbose:
                print("Máscaras de electrodos cargadas.")
        
        self.skeletonize()
        if verbose:
            print("Esqueleto generado.")
            
        self.get_network()
        if verbose:
            print("Red obtenida del esqueleto.")
        
        if clean_network:
            self.clean_network()
            if verbose:
                print("Red limpiada.")
        
        self.convert_to_line_graph()
        if verbose:
            print("Convertida a grafo de líneas.")
        
        if save_graph_path:
            self.save_graph(save_graph_path)
            if verbose:
                print(f"Grafo guardado en {save_graph_path}.")
        
        if plot:
            self.plot_graph(save_plot_path)
            if verbose:
                print("Gráfico generado.")
        
        return
    
    