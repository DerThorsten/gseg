import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv


import gseg



app = QtGui.QApplication([])














visu 	 = False
filepath = 'cell.jpg'
img		 = numpy.squeeze(vigra.readImage(filepath))
gradmag  = vigra.filters.gaussianGradientMagnitude(img,4.0)

seg,nseg = vigra.analysis.watersheds(gradmag)







if visu:
	app = QtGui.QApplication([])
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


eGlobal=gseg.energy_functions.ReconstructionError(image=img,cgp=cgp,beta=0.5,norm=2)


# some labeling


def run():

	for x in range(500):
		primalLabeling = numpy.random.randint(2, size=nCells2)
		eg=eGlobal(argPrimal=primalLabeling)
		if x%10==0:
			print eg


import cProfile 
cProfile.run("run()")



# make label image from that


# segmentation layer




print nCells0,nCells1,nCells2




