






def saveToFile(
	imageNames
	inputSetting,
	parameters,
	outputSetting,
	f,
	overwrite = True

):
	""" store function results to file

	kwargs :
		fil
		inputFiles  : a list of full h5 filepath

	"""
	
	pass




if __name__ == "__main__":

	# a watershed example

	saveToFile(
		imageNames    		= ["bear1","bear2"] , 
		inputSetting  		= dict(seedImage=("/seed_image/","img"),growImage=("/grow_image/","img")),
		parameters 	  		= dict(size=10,beta=0.5),
		outputSetting      	= [ ("/seg_ws/","seg") ,("/nseg/,nseg")   ],
		f 					= vigra.filters.watershed,
		overwrite			= False
 	)