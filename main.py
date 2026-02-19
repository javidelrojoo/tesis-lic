from SkeletonAnalysis import SkeletonAnalysis

muestra = 'M3'
dir_path = './muestras/' + muestra + '/'
img_path = dir_path + 'M3_binary.tif'
electrodes_path = (dir_path + 'M3_electrodo_mask1.JPG', dir_path + 'M3_electrodo_mask2.JPG')

SA = SkeletonAnalysis(img_path)

SA.complete_analysis(electrodes_path=electrodes_path, plot=True, verbose=True)