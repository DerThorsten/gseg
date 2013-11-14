from sklearn.cluster import Ward,WardAgglomeration
import numpy
import phist
import opengm

class CgpClustering(object):
	def __init__(self,cgp):
		self.cgp = cgp 
		self.labels    = numpy.zeros(self.cgp.numCells(2),dtype=numpy.uint64)


		

class HierarchicalClustering(CgpClustering):
	def __init__(self,cgp):
		super(HierarchicalClustering, self).__init__(cgp)
		self.connectivity 	= cgp.sparseAdjacencyMatrix()

	def segment(self,features,nClusters):

		#print "features",features.shape
		#print "self.connectivity",self.connectivity.shape

		self.ward = WardAgglomeration(n_clusters=nClusters, connectivity=self.connectivity).fit(features.T)
		self.labels[:] = self.ward.labels_




class HierarchicalLabelHistogramClustering(object):
	def __init__(self,cgp):
		super(HierarchicalLabelHistogramClustering, self).__init__(cgp)
		self.hCluster  = HierarchicalClustering(cgp)

	def segment(self,pixelLabels,nLabels,nClusters,r=1,sigma=1.0):

		# get the label histograms (x,y,seg,label_bin)
		labelHist = phist.labelHistogram(img=pixelLabels,nLabels=nLabels,r=r,sigma=sigma)
		# reshape to a "default"feature image (x,y,feature)
		labelHist = labelHist.reshape([pixelLabels.shape[0],pixelLabels.shape[1],-1])

		# cluster with HierarchicalClustering
		self.hCluster.segment(features=labelHist,nClusters=nClusters)
		self.labels[:] = self.hCluster.labels



"""

class ColorHistKMeanFusionClustering(object):
	def __init__(self,cgp):
		super(HierarchicalLabelHistogramClustering, self).__init__(cgp)

		self.pixelLabels = None

	def segment(self,images,joint=False,binsColorHist=32,rColorHist=1,sigmaColorHist=[1.0,2.0],k):

		if joint=False :

			nChannels    = pixelFeatures.shape[2]
			nColorSpaces = nChannels / 3

			# allocate space for histogram
			histShape = [pixelFeatures.shape[0],pixelFeatures.shape[1],3,bins]
			histogram = numpy.zeros(histShape ,dtype=numpy.float32)

			for imgIndex, img in enumerate(image):

				# compute the histogram for that color space
				histogram = phist.histogram(image=img,bins=binsColorHist,sigma=sigmaColorHist,r=binsColorHist)

"""







class MulticutClustering(CgpClustering):
	def __init__(self,cgp):
		super(MulticutClustering, self).__init__(cgp)

		# build a graphical model 
		nVar 	    = cgp.numCells(2)
		nFac 		= cgp.numCells(1)
		cell1Bounds = cgp.cell1BoundsArray()-1

		self.gm = opengm.gm(numpy.ones(nVar,dtype=opengm.label_type)*nVar)

		# init with zero potts functions
		fids = self.gm.addFunctions(opengm.pottsFunctions([nVar,nVar],numpy.zeros(nFac),numpy.zeros(nFac) ))
		# add factors 
		self.gm.addFactors(fids,cell1Bounds)

		self.cgc = opengm.inference.Cgc(gm=self.gm,parameter=opengm.InfParam(planar=True)) 

	def segment(self,weights,warmStart=None,verbose=False):

		#try :
		#	self.cgc.changeWeights(weights)

		#except :
		nVar 	    = self.cgp.numCells(2)
		nFac 		= self.cgp.numCells(1)
		cell1Bounds = self.cgp.cell1BoundsArray()-1
		self.gm2 = opengm.gm(numpy.ones(nVar,dtype=opengm.label_type)*nVar)

		assert self.gm2.numberOfVariables == nVar
		# init with zero potts functions
		#print weights

		pf = opengm.pottsFunctions([nVar,nVar],numpy.zeros(nFac),weights )
		fids = self.gm2.addFunctions(pf)


		# add factors 
		self.gm2.addFactors(fids,cell1Bounds)

		self.cgc2 = opengm.inference.Cgc(gm=self.gm2,parameter=opengm.InfParam(planar=True)) 



		if verbose :
			self.cgc2.infer(self.cgc.verboseVisitor())
		else :
			self.cgc2.infer()
		
		self.labels[:]=self.cgc2.arg()



def multicutClustering(cgp,weights,verbose=False):
	segmentor = MulticutClustering(cgp)
	segmentor.segment(weights,verbose=verbose)
	return segmentor.labels