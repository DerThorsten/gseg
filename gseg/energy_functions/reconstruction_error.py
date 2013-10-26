import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv



class ReconstructionError(object):
	def __init__(self,image,cgp,beta,norm=2):
		self.cgp   = cgp
		self.image = numpy.array(image) 
		self.beta  = beta
		self.norm  = norm


	def __call__(self,argPrimal,argDual=None):


		# get feature image
		labelImage = self.cgp.featureToImage(
			cellType=2,
			features=argPrimal.astype(numpy.float32),
			ignoreInactive=True,
			useTopologicalShape=False
		)



		denseLabels = vigra.analysis.labelImage(labelImage)

		#print "limg",labelImage.min(),labelImage.max()
		#print "dimg",denseLabels.min(),denseLabels.max()


		#print "get new cgp"
		tgrid2 	= cgp2d.TopologicalGrid(denseLabels.astype(numpy.uint64))
		cgp2  	= cgp2d.Cgp(tgrid2)


		fdict,activeVals  = cgp2.accumulateCellFeatures(2,image=self.image,features='Mean')

		mean=fdict['Mean'].astype(numpy.float32)

		#print "meanshape ",mean.shape
		#print "meandtype ",mean.dtype
		meanImage = numpy.array(cgp2.featureToImage(
			cellType=2,
			features=mean,
			ignoreInactive=True,
			useTopologicalShape=False
		))










		# get number of regions
		nSeg      = denseLabels.max() 
		nPixel    = int(self.image.shape[0]*self.image.shape[1])


		diff = self.image[:]-meanImage[:]
		
		normImage = numpy.abs(diff)**self.norm
		normSum   = numpy.sum(normImage)

		# normalize by the number of pixels
		avPixelError = normSum / float( nPixel )



		if False :


			#app = QtGui.QApplication([])
			viewer =  lv.LayerViewer()
			viewer.show()
				
			# input gray layer
			viewer.addLayer(name='LabelImage',layerType='GrayLayer')
			viewer.setLayerData(name='LabelImage',data=labelImage)

			viewer.addLayer(name='Mean',layerType='GrayLayer')
			viewer.setLayerData(name='Mean',data=meanImage)

			viewer.addLayer(name='Norm',layerType='GrayLayer')
			viewer.setLayerData(name='Norm',data=normImage)

			viewer.addLayer(name='denseImage',layerType='SegmentationLayer')
			viewer.setLayerData(name='denseImage',data=denseLabels)


			viewer.autoRange()
			QtGui.QApplication.instance().exec_()


		#print "av px error ",avPixelError , "weighted nSeg",self.beta * float(nSeg)  

		return avPixelError + self.beta * float(nSeg)  
