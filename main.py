from SkeletonAnalysis import SkeletonAnalysis
import matplotlib.pyplot as plt

muestra = 'M1'
dir_path = './muestras/' + muestra + '/'
img_path = dir_path + muestra + '_binary.tif'
electrodes_path = (dir_path + muestra + '_electrode_mask1.tif', dir_path + muestra + '_electrode_mask2.tif')

SA = SkeletonAnalysis(img_path)
# SA.load_electrodes(*electrodes_path)
# SA.skeletonize()
# plt.figure(figsize=(8, 8))
# plt.imshow(SA.img, cmap='grey')
# plt.imshow(SA.skeleton_img, cmap='grey', alpha=0.5)
# plt.axis('off')
# plt.tight_layout()
# # plt.savefig(dir_path + 'M3_skeleton.png', bbox_inches='tight', pad_inches=0, dpi=400)
# plt.show()
SA.complete_analysis(plot=False, verbose=True, save_graph_path=dir_path + muestra + '.graphml', electrodes_path=electrodes_path)