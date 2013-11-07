




class  KHistMean(object):

	def __init__(self,k,histograms,weights):

		self.nBins			   = histograms.shape[1]
		self.centroidHistogram = None
		self.labels 		   = None
		self.nAssigments 	   = None

	def updateAssigments(self):
		# compute distances for
		# each histogram to den centroids


	def updateCentroids(self):
		# merge all histogram 