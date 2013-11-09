import vigra
import numpy
import phist



def colorSpaceDescriptor(imgRgb):
	rgb = imgRgb
	lab  = vigra.colors.transform_RGB2Lab(rgb)	
	luv  = vigra.colors.transform_RGB2Luv(rgb)
	xyz  = vigra.colors.transform_RGB2XYZ(rgb)
	rgbp = vigra.colors.transform_RGB2RGBPrime(rgb)
	cbcr = vigra.colors.transform_RGBPrime2YPrimeCbCr(rgbp)
	return  numpy.concatenate([rgb,lab,luv,xyz,rgbp,cbcr],axis=2)






