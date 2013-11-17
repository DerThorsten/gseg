from sklearn.cluster import MiniBatchKMeans  
import numpy
import vigra
import matplotlib
import pylab



def kMeansPixelColoring(features,k,visu=False):
    """	Call k means for a single feature space.

    kwargs:
        features : at least a 3d arrays .
        The first 2 axis are the x and y axis.
        The next dimensions are the features

        k : number of clusters 

    returns : sigle segmentations
    """
    segmentor = MiniBatchKMeans(k)
    f = features.reshape(features.shape[0]*features.shape[1],-1)
    labels = segmentor.fit_predict(f)
    labels = labels.reshape([features.shape[0],features.shape[1]])

    if visu:
        cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
        pylab.imshow ( numpy.swapaxes(labels,0,1), cmap = cmap)
        pylab.show()


    return labels



def batchKMeanPixelColoring(fetures,k,visu=False):
	"""	Call k means multiple times for different feature spaces.

		kwargs:
			features : at least a 4d arrays .
				The first 2 axis are the x and y axis.
				The next dimension is the "batch" dimension.
				And all following dimensions are bins

			k : number of clusters 

		returns : Multiple segmentations in one array


	"""
	if visu:
		# A random colormap for matplotlib
		cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))


	dx,dy,nBatch= fetures.shape[0:3]
	semiFlat = fetures.reshape([dx,dy,nBatch,-1])
	out = numpy.zeros([dx,dy,nBatch],dtype=numpy.uint64)

	for bi in range(nBatch):
	    hist = semiFlat[:,:,bi,:]
	    labels = kMeansPixelColoring(hist,k)
	    out[:,:,bi]=labels[:,:]

	    if visu:
		    pylab.imshow ( numpy.swapaxes(labels,0,1), cmap = cmap)
		    pylab.show()

	return out

