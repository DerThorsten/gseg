




class ReconstructionError(object):
	def __init__(self,image,cgp,beta,norm=2):
		self.cgp   = cgp
		self.image = image 
		self.beta  = beta
		self.norm  = norm


	def __call__(self,argPrimal,argDual):

		# get number of regions
		nSeg      = argPrimal.max() +1 
		nPixel    = self.image[0]*self.image[1]

		# get mean "color/feature" for each region
		result 	= cgp.accuulateFeatures(cellType=2,image=self.image,accumulators=['mean'])
		mean 	= result['mean']

		# write result back into an actual image
		meanImage = cgp.projetFeatureToImage(cellType=2,topologicalShape=False)
		normImage = numpy.abs(self.image-meanImage)**self.norm
		normSum   = numpy.sum(normImage)

		# normalize by the number of pixels
		avPixelError = normSum / float( nPixel )

		return avPixelError + self.beta * float(nSeg)  

