import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt


import gseg
import pylab
import matplotlib.cm as cm
import matplotlib
import scipy


def toSomeColorSpaces(rgb,b):
	lab  = vigra.colors.transform_RGB2Lab(rgb)	
	luv  = vigra.colors.transform_RGB2Luv(rgb)
	xyz  = vigra.colors.transform_RGB2XYZ(rgb)
	rgbp = vigra.colors.transform_RGB2RGBPrime(rgb)
	cbcr = vigra.colors.transform_RGBPrime2YPrimeCbCr(rgbp)
	


	xv, yv = numpy.meshgrid(numpy.arange(rgb.shape[1]), numpy.arange(rgb.shape[0]))

	xv = b*(numpy.reshape(xv, [xv.shape[0],xv.shape[1],1] ).astype(numpy.float32)/rgb.shape[0])
	yv = b*(numpy.reshape(yv, [yv.shape[0],yv.shape[1],1] ).astype(numpy.float32)/rgb.shape[1])

	fu = numpy.concatenate([xv,yv],axis=2)

	print "foo",fu.shape
	print "xv",xv
	f = numpy.concatenate([lab,luv,xyz,rgbp,cbcr,xv,yv],axis=2)

	print f.shape
	return f





visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)

f = toSomeColorSpaces(img,b=500.0)



print f
f    = numpy.array(f)

#f =  scipy.cluster.vq.whiten(f.T).T

segmentor = gseg.segmentors.KMeanSegmentor(f)

labels    = segmentor(n_clusters=4)


cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 10000,3))

f = pylab.figure()
for n, img in enumerate([img,labels]):
    f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    if n==0:
    	pylab.imshow(numpy.swapaxes(img,0,1)/255.0)
    else :
    	pylab.imshow(numpy.swapaxes(img,0,1),cmap=cmap)
pylab.title('Double image')
pylab.show()
