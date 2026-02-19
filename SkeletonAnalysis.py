import cv2
import numpy as np
from skimage.morphology import remove_small_holes, opening, disk, skeletonize, dilation
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
        self.has_electrodes = True
    
    def img_cleaning(self):
        self.img = remove_small_holes(self.img.astype(bool), area_threshold=30)
        self.img = opening(self.img, footprint=disk(1))
    
    def skeletonize(self):
        self.skeleton_img = skeletonize(self.img)
        if self.has_electrodes:
            self.skeleton_img = self.skeleton_img & ~self.electrode_mask1 & ~self.electrode_mask2
            
            electrode_dil1 = dilation(self.electrode_mask1, disk(5))
            electrode_dil2 = dilation(self.electrode_mask2, disk(5))

            self.contact_points1 = self.skeleton_img & electrode_dil1 & ~self.electrode_mask1 & ~self.electrode_mask2
            self.contact_points2 = self.skeleton_img & electrode_dil2 & ~self.electrode_mask1 & ~self.electrode_mask2

            self.skeleton_img = self.skeleton_img | self.contact_points1 | self.contact_points2
    
    def get_network(self):
        skeleton = Skeleton(self.skeleton_img)
        self.G = skeleton_to_nx(skeleton)
        self.node_coords = np.array(skeleton.coordinates)
        if self.has_electrodes:
            electrode_nodes1 = []
            electrode_nodes2 = []

            for node_id, (y, x) in enumerate(self.node_coords):
                if self.contact_points1[int(y), int(x)]:
                    electrode_nodes1.append(node_id)
                if self.contact_points2[int(y), int(x)]:
                    electrode_nodes2.append(node_id)

            self.G.add_node("Vin", pos=(2400, -1000))
            self.G.add_node("Vout", pos=(3200, -4500))  

            for n in electrode_nodes1:
                self.G.add_edge("Vin", n)

            for n in electrode_nodes2:
                self.G.add_edge("Vout", n)
    
    def clean_network(self):
        changed = True
        while changed:
            changed = False

            # Remove degree 1 nodes iteratively
            deg1_nodes = [n for n, d in self.G.degree() if d == 1]
            if deg1_nodes:
                self.G.remove_nodes_from(deg1_nodes)
                changed = True

            # Collapse degree 2 nodes iteratively
            deg2_nodes = [n for n, d in self.G.degree() if d == 2]
            for n in deg2_nodes:
                neighbors = list(self.G.neighbors(n))
                if len(neighbors) == 2:
                    u, v = neighbors
                    self.G.add_edge(u, v)
                self.G.remove_node(n)
                changed = True

            if any(nx.selfloop_edges(self.G)):
                self.G.remove_edges_from(nx.selfloop_edges(self.G))
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
        for i, n in enumerate(self.G.nodes):
            if n == "Vin":
                pos[n] = (2400, -1000)  # arbitrary position
            elif n == "Vout":
                pos[n] = (3200, -4500)  # arbitrary position
            else:
                y, x = self.node_coords[n]
                pos[n] = (x, y)
        plt.figure(figsize=(8, 8))
        nx.draw(self.G, pos, node_size=25, width=0.8)
        plt.imshow(self.img, cmap='gray')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.show()
    
    def complete_analysis(self, electrodes_path=None, save_graph_path=None, plot=True, save_plot_path=None):
        self.img_cleaning()
        if electrodes_path:
            self.load_electrodes(*electrodes_path)
        self.skeletonize()
        self.get_network()
        self.clean_network()
        self.convert_to_line_graph()
        if save_graph_path:
            self.save_graph(save_graph_path)
        if plot:
            self.plot_graph(save_plot_path)
    
    