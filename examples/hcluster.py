import vigra
import opengm
import cgp2d
import numpy
import gseg
import phist

###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
seg,nseg    = vigra.analysis.slicSuperpixels(imgLab,15.0,5)
seg 		= vigra.analysis.labelImage(seg)
tgrid 		= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  		= cgp2d.Cgp(tgrid)
features    = cgp.accumulateCellFeatures(cellType=2,image=imgLab,features="Mean")[0]['Mean']
imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)




print "compute histograms"
#h0 = phist.jointHistogram(image=img,bins=5,r=3,sigma=None)
h1 = phist.jointHistogram(image=imgLab,bins=5,r=3,sigma=[0.7,1.0])
#h0 = h0.reshape( [h0.shape[0],h0.shape[1],-1 ])
h1= h1.reshape( [h1.shape[0],h1.shape[1],-1 ])
print h1.shape

print "accumulate hist features"
featuresB    = cgp.accumulateCellFeatures(cellType=2,image=h1,features="Mean")[0]['Mean']

###############################################################################
# visualize overseg
###############################################################################
print "number of regions",cgp.numCells(2)
cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)


###############################################################################
# segment
###############################################################################
segmentor = gseg.segmentors.HierarchicalClustering(cgp=cgp)
segmentor.segment(featuresB,100)
labels 	= segmentor.labels 



cell1State = numpy.zeros(cgp.numCells(1))
cell1Bounds=cgp.cell1BoundsArray()-1





for ci  in xrange(cgp.numCells(1)):
	
	r1,r2  = cell1Bounds[ci,:]
	if labels[r1]!=labels[r2]:
		cell1State[ci]=1.0


cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))

