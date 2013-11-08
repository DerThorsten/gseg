import vigra
import opengm
import cgp2d
import numpy
import gseg
import phist

import h5py

from sklearn import preprocessing

import matplotlib
import pylab
# A random colormap for matplotlib
cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))



###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
#filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
seg,nseg    = vigra.analysis.slicSuperpixels(imgLab,15.0,5)
seg 		= vigra.analysis.labelImage(seg)
tgrid 		= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  		= cgp2d.Cgp(tgrid)





dx = img.shape[0]
dy = img.shape[1]




print "get / load pixel colorspace histograms"

if False :
    print "get color spaces"
    cp    	= gseg.features.colorSpaceDescriptor(img)

    nCp = cp.shape[2]/3

    print "get jointColorHistDescriptor"
    jhist = gseg.features.jointColorHistDescriptor(cp)

    #hist = hist.reshape( [hist.shape[0],hist.shape[1],nCp,-1 ]  )



    f = h5py.File('colorHist.h5','w')
    f['cp']   =cp
    f['jhist']=jhist
    f.close()
else : 
    f = h5py.File('colorHist.h5','r')
    cp = f['cp'].value
    jhist = f['jhist'].value
    f.close()





print "compute k means on a signle colorspaec histogram (125 bins)"
k=6
if True :
    nCp = jhist.shape[2]

    print "histoshape ",jhist.shape,"nCp",nCp

    for cp in range(nCp):
        print "cp",cp
        hist = jhist[:,:,cp,:,:,:]
        hist = hist.reshape([dx,dy,-1])


        labels = gseg.segmentors.kMeansColoring(hist,k)

        pylab.imshow ( numpy.swapaxes(labels,0,1), cmap = cmap)
        pylab.show()

else :
    pass









#features    = cgp.accumulateCellFeatures(cellType=2,image=jhist.reshape([dx,dy,-1]),features="Mean")[0]['Mean']
#features = preprocessing.scale(features)
features    = cgp.accumulateCellFeatures(cellType=2,image=imgLab,features="Mean")[0]['Mean']


print "features",features.shape

imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)








###############################################################################
# visualize overseg
###############################################################################
print "number of regions",cgp.numCells(2)
cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)


###############################################################################
# segment
###############################################################################
segmentor = gseg.segmentors.HierarchicalClustering(cgp=cgp)
segmentor.segment(features,100)
labels 	= segmentor.labels 



cell1State = numpy.zeros(cgp.numCells(1))
cell1Bounds=cgp.cell1BoundsArray()-1





for ci  in xrange(cgp.numCells(1)):
	
	r1,r2  = cell1Bounds[ci,:]
	if labels[r1]!=labels[r2]:
		cell1State[ci]=1.0


cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))

