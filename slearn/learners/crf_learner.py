











class CeLearner(object):
	def __init__(
		models,
		featuresFunctions

	):
	"""	Cross Entropy Learning:

		kwargs:
			models :
				encodes the structe sof the models
				and so far this has to be a containers of "cgp's"
				which is a cell complex graph
				( extended region adj. graph )

			featuresFunctions : 
				a list / container of feature functions 
				which  are parametrized by some parameters
				which are learned.


	"""