import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt
import gseg



# Author : Vincent Michel, 2010
#          Alexandre Gramfort, 2011
# License: BSD 3 clause

print(__doc__)

import time as time
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward

###############################################################################
# Generate data

visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
gradmag  	= vigra.filters.gaussianGradientMagnitude(img,4.0)

seg,nseg    = vigra.analysis.slicSuperpixels(imgLab,10.0,5)
seg 		= vigra.analysis.labelImage(seg)


#seg,nseg 	= vigra.analysis.watersheds(gradmag)

print "nseg",nseg

tgrid 	= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  	= cgp2d.Cgp(tgrid)


imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)
gradTopo 	= vigra.filters.gaussianGradientMagnitude(imgTopo,1.0)
labelsTopo  = vigra.sampling.resize(seg.astype(numpy.float32),cgp.shape,0)



print "number of regions",cgp.numCells(2)


# visualize segmetation

cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)#,edge_data_in=bestState.astype(numpy.float32))



f,a=cgp.accumulateCellFeatures(cellType=2,image=imgLab,features="Mean")
connect = cgp.sparseAdjacencyMatrix()

f=f['Mean']

print "features",f.shape
print "connectity",connect.shape










X = f[:,:]

###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = connect


print type(connectivity)

print connectivity



###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 500  # number of regions
ward = Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X)
label = ward.labels_



cell1State = numpy.zeros(cgp.numCells(1))
cell1Bounds=cgp.cell1BoundsArray()-1





for ci  in xrange(cgp.numCells(1)):
	
	r1,r2  = cell1Bounds[ci,:]
	if label[r1]!=label[r2]:
		cell1State[ci]=1.0


cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))


print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

###############################################################################



# Plot the results on an image
pl.figure(figsize=(5, 5))
pl.imshow(lena, cmap=pl.cm.gray)
for l in range(n_clusters):
    pl.contour(label == l, contours=1,
               colors=[pl.cm.spectral(l / float(n_clusters)), ])
pl.xticks(())
pl.yticks(())
pl.show()