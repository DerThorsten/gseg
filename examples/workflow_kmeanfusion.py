import vigra
import opengm
import cgp2d
import numpy
import gseg
import phist
from  lazycall import LazyArrays , LazyCaller,getFiles, makeFullPath
import h5py
from sklearn import preprocessing
import matplotlib
import pylab




n = 200
imagePath   		= "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/"
files ,baseNames 	= getFiles(imagePath,"jpg")
files 				= files[0:n]
baseNames 			= baseNames[0:n]

# rgb image
images      		= LazyArrays(files=files,filetype="image") 
# color space arrays
csp      			= LazyArrays(files=makeFullPath("/home/tbeier/dump",baseNames,"h5"),dset="data",filetype="h5") 



# color space convertion for all files in bsd
batchFunction = LazyCaller(f=gseg.features.colorSpaceDescriptor,verbose=True)
batchFunction.name = "colorpsace conversion"
batchFunction.overwrite=True
batchFunction.compress=True
batchFunction.skipAll =False
batchFunction.compressionOpts=2
batchFunction.setBatchKwargs(["imgRgb"])
batchFunction.setOutput(files=csp.files,dset=csp.dset)
batchFunction(imgRgb=images)


print csp[0].shape
print csp[0]





sys.exit()



# A random colormap for matplotlib
cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))




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

