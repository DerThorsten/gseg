import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt


import gseg









visu 	 	= True
filepath 	= '42049.jpg'
#filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
gradmag  	= vigra.filters.gaussianGradientMagnitude(img,4.0)











def patchNormalization(img)