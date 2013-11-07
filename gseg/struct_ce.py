



"""

	rdy seg features 


		- canny
		- color channel k - mean  - texton kmean
		- does it touch the boarder
		- border dist 





"""







"""

LINEAR

w0           +     w1 * l1  + ... + wN * lN     
{ offset }   +   { weighted features }


QUADRATIC

w0           +   w1_1 * l1^1  + w1_2 * l1^2  + ....... w2_1 * l2^1  + w2_2 * l2^2 


"""



def vi(a,b):
	pass


class TrainingsSet(object):

	def getFeatures(index):
		pass

	def getGts(index):
		pass

	def __len__():
		pass




def learning(
	trainingImages,
	weightToSeg,
	loss, 
	nSamples,
	nEliteSamples
	conv = 0.00000001
):
	
	eliteWeightSamples 
	eliteWeightEnergy
	weightSamples
	while(True):

		# get N weight samples# (nSamples , |W|)
		weightSamples  = weightToSeg.getWeightSamples(nSamples)
		weightSamplesEnergy 

		for imageIndex,trainingImage in enumerate(trainingImages):	


			gts = trainingImages.gts

			# generate #nSamples  segmentations according to the weights
			segmentations = weightToSeg.generateSegmentations(weightSamples)
			# get energies of samples (one energy per weight sample)
			energies 		  =  loss(segmentations,gts)
			weightSamplesEnergy[imageIndex,:]=energies


		# get elite weights for all images
		imgEliteWeights , imgEliteEnergy = getEliteWeihts(nEliteSamples,weightSamples,weightSamplesEnergy)
		eliteWeightSamples[imageIndex,:,:]=imgEliteWeights[:,:]
		eliteWeightEnergy[imageIndex,;]=imgEliteEnergy[:]

		# update weight probability density from eliteWeightSamples

		dist = weightToSeg.updateWeights(eliteWeightSamples,eliteWeightEnergy)

		if dist < conv :
			break