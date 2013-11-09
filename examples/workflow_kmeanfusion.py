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
import phist



n = 25
imagePath   		= "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/"
files ,baseNames 	= getFiles(imagePath,"jpg")
files 				= files[0:n]
baseNames 			= baseNames[0:n]

# rgb image
images      		= LazyArrays(files=files,filetype="image") 
# color space arrays
csp      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/csp",baseNames,"h5"),dset="data",filetype="h5") 
# jhist
jhist      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/jhist",baseNames,"h5"),dset="data",filetype="h5") 
# kmseg
kmseg      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/kmseg",baseNames,"h5"),dset="data",filetype="h5") 

# kmseg
lhist      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/lhist",baseNames,"h5"),dset="data",filetype="h5") 


# color space convertion for all files in bsd
batchFunction = LazyCaller(f=gseg.features.colorSpaceDescriptor,verbose=True)
batchFunction.name = "colorpsace conversion"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["imgRgb"])
batchFunction.setOutput(files=csp.files,dset=csp.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(imgRgb=images)





# compute batch joint histogramm
# (one 5x5x5 = 125 bin histogram for each color channel)
batchFunction = LazyCaller(f=phist.batchJointHistogram,verbose=True)
batchFunction.name = "joint histograms"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["img"])
batchFunction.setOutput(files=jhist.files,dset=jhist.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(img=csp,r=1,bins=5,sigma=[1.0,1.0])




# colorHistKMeans
batchFunction = LazyCaller(f=gseg.segmentors.batchKMeanPixelColoring,verbose=True)
batchFunction.name = "joint hist k-means"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["fetures"])
batchFunction.setOutput(files=kmseg.files,dset=kmseg.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(fetures=jhist,k=15,visu=False)





# label histogram
batchFunction = LazyCaller(f=phist.labelHistogram,verbose=True)
batchFunction.name = "label histogram"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["img"])
batchFunction.setOutput(files=lhist.files,dset=lhist.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(img=kmseg,nLabels=15,r=1,sigma=1.6,visu=False)


print "img here"
for i in range(0,n):
    k=100
    allcsp = csp[i]
    hist   = lhist[i]
    rgb = allcsp[:,:,0:3]/255.0
    lab = allcsp[:,:,3:6]

    #pylab.imshow ( numpy.swapaxes(rgb,0,1))#, cmap = "jet")
    #pylab.show()



    ###############################################################################
    # Generate data
    ###############################################################################

    print "labshape",lab.shape


    img         = rgb
    imgLab      = lab

    imgLab = vigra.VigraArray(imgLab,axistags=vigra.defaultAxistags("xyc"))

    hist = hist.reshape([hist.shape[0],hist.shape[1],-1])

    hist = vigra.VigraArray(hist,axistags=vigra.defaultAxistags("xyc"))
    imgLab = vigra.VigraArray(imgLab,axistags=vigra.defaultAxistags("xyc"))

    fo = numpy.swapaxes(imgLab,0,2)
    fo = numpy.swapaxes(fo,1,2)
    fo = vigra.RGBImage( fo )
    seg,nseg    = vigra.analysis.slicSuperpixels(fo,10.0,5)
    print "done"
    seg         = vigra.analysis.labelImage(seg)
    tgrid       = cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
    cgp         = cgp2d.Cgp(tgrid)



    #features    = cgp.accumulateCellFeatures(cellType=2,image=jhist.reshape([dx,dy,-1]),features="Mean")[0]['Mean']
    #features = preprocessing.scale(features)
    features    = cgp.accumulateCellFeatures(cellType=2,image=hist,features="Mean")[0]['Mean']


    print "features",features.shape

    imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
    imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)
    #cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)





    ###############################################################################
    # segment
    ###############################################################################
    segmentor = gseg.segmentors.HierarchicalClustering(cgp=cgp)
    segmentor.segment(features,k)
    labels 	= segmentor.labels 



    cell1State = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)
    cell1Bounds=cgp.cell1BoundsArray()-1





    for ci  in xrange(cgp.numCells(1)):
    	
    	r1,r2  = cell1Bounds[ci,:]
    	if labels[r1]!=labels[r2]:
    		cell1State[ci]=1


    tgrid2 = cgp.merge2Cells(cell1State)
    cgp2   = cgp2d.Cgp(tgrid2)

    cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))
    cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp2)#,edge_data_in=cell1State.astype(numpy.float32))
