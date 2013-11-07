import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv



class ReconstructionError(object):

	def __init__(self,image,edgeImage,cgp,beta,gamma,norm=2):
		self.cgp   		= cgp
		self.image 		= numpy.array(image) 
		self.edgeImage  = edgeImage
		self.beta  		= beta
		self.norm  		= norm


		# accumulate cell-1 edge indicator
		fdict,activeVals  = self.cgp.accumulateCellFeatures(1,image=self.edgeImage,features='Mean')
		edgeWeights=fdict['Mean'].astype(numpy.float32)

		e1 = numpy.exp(-1.0*gamma* edgeWeights)
		e0 = 1.0 -e1

		self.weights = e1-e0

		print edgeWeights.shape
		print "g",edgeWeights[0:5]
		print "0",e0[0:5]
		print "1",e1[0:5]
		print "w",self.weights[0:5]

	def __call__(self,argPrimal,argDual=None):
		where1=numpy.where(argDual==1)
		costs = numpy.sum(self.weights[where1])
		return costs


