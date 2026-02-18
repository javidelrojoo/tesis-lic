import cv2
import numpy as np
from skimage.morphology import remove_small_holes, opening, disk, skeletonize, dilation
from skan import Skeleton
from skan.csr import skeleton_to_nx
import networkx as nx

class SkeletonAnalysis:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.has_electrodes = False
    
    def load_electrodes(self, path1, path2):
        self.electrode_mask1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        self.electrode_mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        self.has_electrodes = True
    
    def img_cleaning(self):
        self.img = remove_small_holes(self.img.astype(bool), max_size=30)
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
        
        if self.has_electrodes:
            node_coords = np.array(skeleton.coordinates)
            electrode_nodes1 = []
            electrode_nodes2 = []

            for node_id, (y, x) in enumerate(node_coords):
                if self.contact_points1[int(y), int(x)]:
                    electrode_nodes1.append(node_id)
                if self.contact_points2[int(y), int(x)]:
                    electrode_nodes2.append(node_id)

            self.G.add_node("Vin", pos=(2400, -1000))  # arbitrary position if you plot later
            self.G.add_node("Vout", pos=(3200, -4500))  

            for n in electrode_nodes1:
                self.G.add_edge("Vin", n)

            for n in electrode_nodes2:
                self.G.add_edge("Vout", n)
    
    def convert2line_graph(self):
        self.L = nx.line_graph(self.G)
        self.L = nx.Graph(self.L)
    
    