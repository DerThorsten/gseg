from sklearn.cluster import MiniBatchKMeans  
import numpy
import vigra

class PixelSegmentor(object):
	def __init__(self,*arg,**kwargs):
		pass




class KMeanSegmentor(PixelSegmentor):
	def __init__(self,features):
		self.shape  	= [features.shape[0],features.shape[1]]
		self.nPixel   	= self.shape[0]*self.shape[1]
		self.nFeatures 	= features.shape[2]
		self.features 	= features.reshape([self.nPixel,self.nFeatures])
		self.labels   	= numpy.zeros(self.nPixel,dtype=numpy.uint32)

	def __call__(self,**kwargs):

		# run  k- means
		segmentor 	   = MiniBatchKMeans(**kwargs)
		self.labels[:] = segmentor.fit_predict(self.features)


		# make a labeled image from 
		notDenseLabels  = self.labels.reshape(self.shape)
		denseLabels 	= vigra.analysis.labelImage(notDenseLabels)

		return denseLabels



