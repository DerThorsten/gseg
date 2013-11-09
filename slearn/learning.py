



class Learning(object):
	def __init__(
		models,					
		featureFunctions,							
		gts,					
		lossFunction, 			
		learner,				
		structureRealizer,		
	):
		"""
			models					: a list of cgps  for each image in training set 

			featureFunctions		: a list of feature functions for each image in training set 
									  a single feature function may return multiple features 
									  And can also give a loss for a given set of parameters 

			gts 					: a maybe nested list of gts for each image in the training set 

			lossFunction  			: fuction returning a loss given a gt and a proposal segmentation 

			learner   				: the learner which should be used atm just cross entropy 

			structureRealizer		: a class which can generate a structe from a single weight per atomic unit like multicut 

		"""	


		maxIterations = 1000 
		epsilon 	  = 0.000000001
		nModels		  = len(models)

		# get the initial parameters and pass them to 
		# the learner
		parameterSet  = featureFunctions.initalParameter()
		leaner.setParameters(parameterSet)

		# best parameter and it's loss
		bestParameter =  parameterSet
		bestLoss 	  =  float('inf')


		# learning iterations
		for iteration in maxIterations : 

			# get samples of paramters from leaner
			parameterSet = learner.getParameterSamples()
			nSamples 	 = len(parameterSet)

			# allocate loss function 
			loss = numpy.zeros(nModels,nSamples)


			# model iterations 
			for modelIndex,(model,featureFunction,gt) in enumerate(zip(models,featureFunctions,gts)):

				# get get weights from feature function according
				# to the current set of parameter samples
				# weights is a NSamples x NAtomics array
				weightSamples  = featureFunction(parameterSet)

				# get structures from structureRealizer
				# - in the unsupervised segmentation case (only supported)
				#   this will return a set segmentation given atomic weights samples
				structures  = structureRealizer.realize(weightSamples)

				# get the loss from the strucutes
				# - something like variation of information
				# will be NSamples  Vector 
				loss[modelIndex,:] = lossFunction.evaluate(strucutes=structes,gt=gt)




			# pass loss  parameter Set to learner to get new weightSamples from learner
			bestParameter,bestLoss,delta = learner.update(parameterSet=parameterSet,loss=loss)


			if delta < epsilon :
				break


		return bestParameter,bestLoss






