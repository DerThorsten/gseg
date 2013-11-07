import vigra
import opengm
#mport cgp2d

import numpy
import matplotlib.pyplot as plt


#import gseg
import pylab
import matplotlib.cm as cm
import matplotlib
import scipy

import gseg
#
import scipy.ndimage

filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:100,0:100,:]
imgshape    = img.shape[0:2]


bins = (5,		5,		5)
hmin = (0.0 ,	0.0, 	0.0)
hmax = (255.0,	255.0,	255.0)





hshape    = imgshape + bins 
histogram = numpy.zeros(hshape , dtype=numpy.float32)

print "hshape",hshape

#convert each pixel value to an index

index = img.copy()

for d in range(len(bins)):
	index[:,:,d]-=hmin[d]
	index[:,:,d]/=(hmax[d]-hmin[0])
	index[:,:,d]*=(bins[d]-1)


index=index.astype(numpy.uint32)

print index[0,0,:]



for x in xrange(imgshape[0]):
	for y in xrange(imgshape[1]):
		hi = index[x,y,:]
		histogram[x,y,index[x,y,0] , index[x,y,1] ,index[x,y,2]  ]+=1.0



print numpy.where(histogram<1)
	


sigmas = (2,2,0.7,0.7,0.7)

print "colvolve"

smoothed = scipy.ndimage.filters.gaussian_filter(histogram,sigma=sigmas,order=0,mode='constant',cval=0.0)

features = smoothed.reshape( imgshape+ (-1,))

rgb=img

xv, yv = numpy.meshgrid(numpy.arange(rgb.shape[1]), numpy.arange(rgb.shape[0]))


b=1.0
xv = b*(numpy.reshape(xv, [xv.shape[0],xv.shape[1],1] ).astype(numpy.float32)/rgb.shape[0])
yv = b*(numpy.reshape(yv, [yv.shape[0],yv.shape[1],1] ).astype(numpy.float32)/rgb.shape[1])


features = numpy.concatenate([features,xv,yv],axis=2)


segmentor = gseg.segmentors.KMeanSegmentor(features)
labels    = segmentor(n_clusters=200)

cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 10000,3))

f = pylab.figure()
for n, toshow in enumerate([img,labels]):
    f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    if n==0:
    	pylab.imshow(numpy.swapaxes(toshow,0,1)/255.0)
    else :
    	pylab.imshow(numpy.swapaxes(toshow,0,1),cmap=cmap)
pylab.title('Double image')
pylab.show()




for r in range(bins[0]):
	for g in range(bins[1]):
		for b in range(bins[2]):

			f = pylab.figure()
			for n, toshow in enumerate([    img,histogram[:,:,r,g,b] ,smoothed[:,:,r,g,b]   ]):
			    f.add_subplot(3, 1, n)  # this line outputs images on top of each other
			    # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
			    if n==0:
			    	pylab.imshow(numpy.swapaxes(toshow,0,1)/255.0)
			    else :
			    	toshow-=toshow.min()
			    	toshow/=toshow.max()
			    	pylab.imshow(numpy.swapaxes(toshow,0,1))

			pylab.title('Double image')
			pylab.show()