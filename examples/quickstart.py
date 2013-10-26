import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv


import gseg


















visu 	 = False
filepath = 'cell.jpg'
img = vigra.readImage(filepath)
print img.shape
img		 = numpy.squeeze(img)[0:60,0:100]
gradmag  = vigra.filters.gaussianGradientMagnitude(img,2.0)

seg,nseg = vigra.analysis.watersheds(gradmag)







if visu:
	viewer =  lv.LayerViewer()
	viewer.show()

	# input gray layer
	viewer.addLayer(name='Input',layerType='GrayLayer')
	viewer.setLayerData(name='Input',data=img)

	# gradmag gray layer
	viewer.addLayer(name='GradMag',layerType='GrayLayer')
	viewer.setLayerData(name='GradMag',data=gradmag)

	# seg layer
	viewer.addLayer(name='SuperPixels',layerType='SegmentationLayer')
	viewer.setLayerData(name='SuperPixels',data=seg)


	viewer.autoRange()
	QtGui.QApplication.instance().exec_()


tgrid 	= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  	= cgp2d.Cgp(tgrid)
nCells0 = cgp.numCells(0)
nCells1 = cgp.numCells(1)
nCells2 = cgp.numCells(2)



# energy function
eGlobal=gseg.energy_functions.ReconstructionError(image=img,cgp=cgp,beta=0.1,norm=2)

# segmentation oracle
oracle=gseg.oracles.MulticutOracle(cgp=cgp)



# some labeling


def run():

	gseg.optimizer(cgp=cgp,eGlobal=eGlobal,oracle=oracle,initStd=0.7,damping=0.7,img=img)

	
run()

#import cProfile 
#cProfile.run("run()")



# make label image from that


# segmentation layer




