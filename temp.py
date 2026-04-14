import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2


#%% Adaptive Binarization
img_path = 'muestras/M4/M4.tif'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

BLOCK_SIZE = 151  # debe ser impar
C = 15
INTENSITY_THRESH = 55

pad = BLOCK_SIZE // 2
img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

mask_adaptive_padded = cv2.adaptiveThreshold(img_padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, BLOCK_SIZE, C)

mask_adaptive = mask_adaptive_padded[pad:-pad, pad:-pad]

mask_adaptive[img < INTENSITY_THRESH] = 0

def crear_superposicion_color(base_gris, mascara_binaria, color_rgb=[255, 0, 0]):
    base_rgb = cv2.cvtColor(base_gris, cv2.COLOR_GRAY2RGB)
    es_objeto = mascara_binaria > 127
    base_rgb[es_objeto] = color_rgb
    return base_rgb

superposicion_adaptive = crear_superposicion_color(img, mask_adaptive, [220, 20, 60])


plt.close('all')
plt.figure(figsize=(12, 6))

plt.imshow(superposicion_adaptive)
plt.title(f'Superposicion: Adaptativo Gaussiano (block={BLOCK_SIZE}, C={C})\n(Rojo = Binarizado)')
plt.axis('off')

plt.tight_layout()
plt.show()

# cv2.imwrite('muestras/M1/M1_binary_adaptive.tif', mask_adaptive)

#%% Node Merger
path = 'muestras/M3/M30007.graphml'
G = nx.Graph(nx.read_graphml(path))

edges = list(G.edges())
np.random.shuffle(edges)
merge_prob = 0.01

print(f"{G.number_of_nodes()} nodos inciales.")
count_merges = 0
for u, v in edges:
    if G.has_node(u) and G.has_node(v):    
        if np.random.rand() < merge_prob:
            nx.contracted_nodes(G, u, v, self_loops=False, copy=False)
            count_merges += 1
            
print(f"{G.number_of_nodes()} nodos restantes. Se realizaron {count_merges} fusiones.")
#%% Edge Merger
G = nx.Graph(nx.read_graphml(path))
pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), seed=10)

plt.figure()
nx.draw(G, pos)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()

ohmic_ratio = 0.1

nodes = list(G.nodes())

for node in nodes:
    if np.random.rand() < ohmic_ratio:
        
        neighbors = list(G.neighbors(node))
        
        edges_to_add = itertools.combinations(neighbors, 2)
        
        G.add_edges_from(edges_to_add)
        
        G.remove_node(node)

plt.figure()
nx.draw(G, pos)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()