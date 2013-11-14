import vigra
import cgp2d 
import numpy 
from ..segmentors import HierarchicalClustering ,multicutClustering
from sklearn import preprocessing



def nifty_sp(
    imgCsp,
    s=3,
    sf=100.0,
    visu=False

):
   
    imgCsp = vigra.taggedView(imgCsp,axistags=vigra.defaultAxistags("xyc"))
    imgRGB = imgCsp[:,:,0:3]
    imgLab = imgCsp[:,:,3:6]

    dxx = vigra.filters.hessianOfGaussian(imgLab[:,:,0],sigma=1.0)

    features = imgCsp.reshape([imgRGB.shape[0],imgRGB.shape[1],-1])
    features = numpy.concatenate([features,dxx],axis=2)
    feat = vigra.taggedView(features,axistags=vigra.defaultAxistags("xyc"))

    
    #print "feat shape",feat.shape
    seg,nseg    = vigra.analysis.slicSuperpixels(feat,sf,s)
    labels      = vigra.analysis.labelImage(seg)

    #print "%d superpixels" % numseg

    #print "get init cgp and resample image"
    #print "numseg",numseg,labels.min(),labels.max()
    cgp,grid=cgp2d.cgpFromLabels(labels.astype(numpy.uint64))
    imgRGBBig = vigra.sampling.resize(imgRGB,cgp.shape,0)
    if visu:
        cgp2d.visualize(imgRGBBig,cgp)

    #print "numRegions",cgp.numCells(2)

    return labels
    



def stage1Reducer(
    imgCsp,
    overseg,
    lhist,
    visu=False
):
   
    imgCsp = vigra.taggedView(imgCsp,axistags=vigra.defaultAxistags("xyc"))
    imgRGB = imgCsp[:,:,0:3]
    imgLab = imgCsp[:,:,3:6]

    lhist  = lhist.reshape( [imgCsp.shape[0],imgCsp.shape[1] ,-1] )
    overseg = numpy.squeeze(overseg)
    dxx = vigra.filters.hessianOfGaussian(imgLab[:,:,0],sigma=1.0)

    features = imgCsp.reshape([imgRGB.shape[0],imgRGB.shape[1],-1])
    features = numpy.concatenate([features,dxx],axis=2)
    feat = vigra.taggedView(features,axistags=vigra.defaultAxistags("xyc"))

    #print "feat shape",feat.shape
    #print overseg.shape
    labels      = vigra.taggedView(overseg,axistags=vigra.defaultAxistags("xy"))

    #print "%d superpixels" % numseg

    #print "get init cgp and resample image"
    #print "numseg",numseg,labels.min(),labels.max()
    cgp,grid=cgp2d.cgpFromLabels(labels.astype(numpy.uint64))
    imgRGBBig = vigra.sampling.resize(imgRGB,cgp.shape,0)
        

    #print "numRegions",cgp.numCells(2)





    segmentor = HierarchicalClustering(cgp=cgp)
    # whiten the features

    total = numpy.concatenate([feat,lhist],axis=2)


    features    = cgp.accumulateCellFeatures(cellType=2,image=total,features="Mean")[0]['Mean']
    features=preprocessing.scale(features)
    segmentor.segment(features,3000)
    labels  = segmentor.labels 
    cell1State = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)
    cell1Bounds=cgp.cell1BoundsArray()-1

    for ci  in xrange(cgp.numCells(1)):
        
        r1,r2  = cell1Bounds[ci,:]
        if labels[r1]!=labels[r2]:
            cell1State[ci]=1
    

    tgrid2 = cgp.merge2Cells(cell1State)
    cgp2   = cgp2d.Cgp(tgrid2)
    if visu:
        cgp2d.visualize(imgRGBBig,cgp)
        cgp2d.visualize(imgRGBBig,cgp2)

    labeling = cgp2.labelGrid(2,False)
    #print "internal labeling shape",labeling.shape
    return labeling


def stage2Reducer(
    imgCsp,
    overseg,
    visu=False
):
   
    imgCsp = vigra.taggedView(imgCsp,axistags=vigra.defaultAxistags("xyc"))
    imgRGB = imgCsp[:,:,0:3]
    imgLab = imgCsp[:,:,3:6]

    cgp,grid=cgp2d.cgpFromLabels(overseg.astype(numpy.uint64))



    # final multicut stage
    sigmaMC  = 1.5
    gammaMC    = 1.0
    gradMag       = vigra.filters.gaussianGradientMagnitude(imgLab,sigma=1.0)
    gradMag       = numpy.squeeze(gradMag)
    meanGrad      = cgp.accumulateCellFeatures(cellType=1,image=gradMag,features="Mean")[0]['Mean']
    e1 = numpy.exp(-gammaMC*meanGrad)
    e0 = 1.0 - e1
    weights = e1-e0

    #print"weightshape",weights.shape,"ncell1",cgp.numCells(1)

    labels=multicutClustering(cgp,weights,verbose=False)


    cell1State = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)
    cell1Bounds=cgp.cell1BoundsArray()-1

    for ci  in xrange(cgp.numCells(1)):
        r1,r2  = cell1Bounds[ci,:]
        if labels[r1]!=labels[r2]:
            cell1State[ci]=1

    


    tgrid2 = cgp.merge2Cells(cell1State)
    cgp2   = cgp2d.Cgp(tgrid2)
    if visu:
        imgRGBBig=vigra.sampling.resize(imgRGB,cgp.shape,0)
        cgp2d.visualize(img_rgb=imgRGBBig,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32),cmap="jet")
        cgp2d.visualize(imgRGBBig,cgp2)

    return cgp2.labelGrid(2,False)
