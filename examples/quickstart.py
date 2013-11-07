import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv


import gseg


















visu 	 	= True
filepath 	= 'l.jpg'
img 		= vigra.readImage(filepath)[0:200,0:200,:]

gradmag  	= vigra.filters.gaussianGradientMagnitude(img,4.0)
seg,nseg 	= vigra.analysis.watersheds(gradmag)


tgrid 	= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  	= cgp2d.Cgp(tgrid)
nCells0 = cgp.numCells(0)
nCells1 = cgp.numCells(1)
nCells2 = cgp.numCells(2)



imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)
gradTopo 	= vigra.filters.gaussianGradientMagnitude(imgTopo,1.0)
labelsTopo  = vigra.sampling.resize(seg.astype(numpy.float32),cgp.shape,0)




if visu:
	app = QtGui.QApplication([])

	viewer =  lv.LayerViewer()
	viewer.show()

	# input gray layer
	viewer.addLayer(name='Input',layerType='RgbLayer')
	viewer.setLayerData(name='Input',data=imgTopo)

	# gradmag gray layer
	viewer.addLayer(name='GradMag',layerType='GrayLayer')
	viewer.setLayerData(name='GradMag',data=gradTopo)

	# seg layer
	viewer.addLayer(name='SuperPixels',layerType='SegmentationLayer')
	viewer.setLayerData(name='SuperPixels',data=labelsTopo)


	viewer.autoRange()
	QtGui.QApplication.instance().exec_()





# energy function
eGlobal=gseg.energy_functions.ReconstructionError(image=imgTopo,edgeImage=gradTopo,
 cgp=cgp,beta=0.1,gamma=0.3,norm=2)

print "get oracle"
# segmentation oracle
oracle=gseg.oracles.MulticutOracle(cgp=cgp,beta=0.5)



# some labeling


def run():

	gseg.optimizer(cgp=cgp,eGlobal=eGlobal,oracle=oracle,initStd=3.0,damping=0.01,img=imgRGBTopo)

	
run()

#import cProfile 
#cProfile.run("run()")



# make label image from that


# segmentation layer




