from SkeletonAnalysis import SkeletonAnalysis

muestra = 'M3'
dir_path = './muestras/' + muestra + '/'
img_path = dir_path + 'M30007_binary.tif'
electrodes_path = (dir_path + 'M30007_electrode1.tif', dir_path + 'M30007_electrode2.tif')

SA = SkeletonAnalysis(img_path)

SA.complete_analysis(plot=True, verbose=True, save_graph_path=dir_path + 'M30007.graphml', electrodes_path=electrodes_path)