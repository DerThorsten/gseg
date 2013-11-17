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
import matplotlib.pyplot as plt
import pylab
import scipy.ndimage
from sklearn.decomposition import PCA, KernelPCA,TruncatedSVD,NMF,ProjectedGradientNMF,RandomizedPCA,SparsePCA

def flip(img):
    img=numpy.squeeze(img)
    return numpy.swapaxes(img,0,1)


def showlab(imgLab):
    rgb=vigra.colors.transform_Lab2RGB(imgLab)
    rgb-=rgb.min()
    rgb/=rgb.max()

    plt.imshow(flip(rgb))
    plt.show()


n = 5
imagePath   		= "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/"
files ,baseNames 	= getFiles(imagePath,"jpg")
files 				= files[0:n]
baseNames 			= baseNames[0:n]

# rgb image
images      		= LazyArrays(files=files,filetype="image") 


# color space arrays
csp      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/csp",baseNames,"h5"),dset="data",filetype="h5") 

# color space arrays
sift                 = LazyArrays(files=makeFullPath("/home/tbeier/dump/sift",baseNames,"h5"),dset="data",filetype="h5") 

# jhist
jhist      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/jhist",baseNames,"h5"),dset="data",filetype="h5") 
# kmseg
kmseg      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/kmseg",baseNames,"h5"),dset="data",filetype="h5") 

# kmseg
lhist      			= LazyArrays(files=makeFullPath("/home/tbeier/dump/lhist",baseNames,"h5"),dset="data",filetype="h5") 

# oversegmentation
oseg                = LazyArrays(files=makeFullPath("/home/tbeier/dump/overseg",baseNames,"h5"),dset="data",filetype="h5") 

# oversegmentation
oseg1                = LazyArrays(files=makeFullPath("/home/tbeier/dump/overseg1",baseNames,"h5"),dset="data",filetype="h5") 

# oversegmentation
oseg2                = LazyArrays(files=makeFullPath("/home/tbeier/dump/overseg2",baseNames,"h5"),dset="data",filetype="h5")




def reduceMe(features):
    #features=features[:,:,0:3,:,:,:]


    print "supergrad"

    features = features.reshape(features.shape[0],features.shape[1],-1)
    features = vigra.taggedView(features,axistags=vigra.defaultAxistags("xyc"))
    gradmagf = vigra.filters.gaussianGradientMagnitude(features,sigma=1.0)[:,:,0]
    gradmagc = numpy.zeros([features.shape[0],features.shape[1],features.shape[2]])

    for c in range(features.shape[2]):
        print "channel",c
        gradmagc[:,:,c]=vigra.filters.gaussianGradientMagnitude(features[:,:,c],sigma=1.0)
    features = numpy.array(features)
    gradmagm = numpy.mean(gradmagc,axis=2)
    
    

    f = pylab.figure()
    for n, img in enumerate([gradmagf,gradmagm]):
        #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
        f.add_subplot(1, 2, n)  # this line outputs images side-by-side
        pylab.imshow(numpy.swapaxes(img,0,1))
    pylab.show()



    print features.shape
    nf=features.shape[2]

    flat        = features.reshape(features.shape[0]*features.shape[1],nf)
    #kpca = PCA(n_components=3, copy=True, whiten=True)
    kpca =SparsePCA(n_components=3)
    #kpca = TruncatedSVD(n_components=3)
    #kpca = ProjectedGradientNMF(n_components=3)
    #kpca = NMF(n_components=3)
    #pca = RandomizedPCA(n_components=3)
    print "do the job"
    X_kpca = kpca.fit_transform(flat)
    #X_back = kpca.inverse_transform(X_kpca)

    print X_kpca.shape

    X_kpca = X_kpca.reshape([features.shape[0],features.shape[1],-1])

    fimg = X_kpca[:,:,0:3].astype(numpy.float32)
    fimg-=fimg.min()
    fimg/=fimg.max()
    fimg = vigra.taggedView(fimg,axistags=vigra.defaultAxistags("xyc"))
    print "fimg",fimg.shape
    gradmag = vigra.filters.gaussianGradientMagnitude(fimg,sigma=2.5)[:,:,0]

    f = pylab.figure()
    for n, img in enumerate([gradmagm,gradmagf,gradmag,fimg]):
        #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
        f.add_subplot(2, 2, n)  # this line outputs images side-by-side
        pylab.imshow(numpy.swapaxes(img,0,1))
    pylab.show()


    def norm(a):

        b=numpy.array(a)
        b-=b.min()
        b/=b.max()
        return b

    supergradmag = norm(0.2*norm(gradmagm)+0.1*norm(gradmagf)+norm(gradmag))

    f = pylab.figure()
    for n, img in enumerate([supergradmag]):
        #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
        f.add_subplot(1, 1, n)  # this line outputs images side-by-side
        pylab.imshow(numpy.swapaxes(img,0,1))
    pylab.show()





    return fimg




#########################################
#           FUN 
#########################################

# oversegmentation
trash                = LazyArrays(files=makeFullPath("/home/tbeier/dump/trash",baseNames,"h5"),dset="data",filetype="h5")




# color space convertion for all files in bsd
batchFunction = LazyCaller(f=gseg.features.denseSift,verbose=True)
batchFunction.name = "dense sift"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["imgIn"])
batchFunction.setOutput(files=sift.files,dset=sift.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(imgIn=images,visu=False)




# color space convertion for all files in bsd
batchFunction = LazyCaller(f=gseg.segmentors.kMeansPixelColoring,verbose=True)
batchFunction.name = "sift k means"
batchFunction.overwrite=True
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["features"])
batchFunction.setOutput(files=trash.files,dset=trash.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(features=sift,k=2,visu=True)













#########################################
#           PIPELINE 
#########################################
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



# oversegmentation 
batchFunction = LazyCaller(f=gseg.segmentors.nifty_sp,verbose=True)
batchFunction.name = "slic overseg"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["imgCsp"])
batchFunction.setOutput(files=oseg.files,dset=oseg.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(imgCsp=csp,s=3,sf=100.0,visu=False)



# stage1reducer
batchFunction = LazyCaller(f=gseg.segmentors.stage1Reducer,verbose=True)
batchFunction.name = "stage 1 reducer"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["imgCsp","overseg","lhist"])
batchFunction.setOutput(files=oseg1.files,dset=oseg1.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(imgCsp=csp,overseg=oseg,lhist=lhist,visu=False)



# stage3reducer
batchFunction = LazyCaller(f=gseg.segmentors.stage2Reducer,verbose=True)
batchFunction.name = "stage 2 reducer"
batchFunction.overwrite=False
batchFunction.skipAll =False
batchFunction.setBatchKwargs(["imgCsp","overseg"])
batchFunction.setOutput(files=oseg2.files,dset=oseg2.dset)
batchFunction.setCompression(True,2)
# DO THE CALL
batchFunction(imgCsp=csp,overseg=oseg1,visu=False)


print "img here"
for i in range(0,n):
    k=10
    allcsp = csp[i]
    hist   = lhist[i]
    rgb = allcsp[:,:,0:3]/255.0
    lab = allcsp[:,:,3:6]
    labeling = oseg2[i]
    print labeling.shape
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
    allcsp = vigra.VigraArray(allcsp,axistags=vigra.defaultAxistags("xyc"))
    imgLab = vigra.VigraArray(imgLab,axistags=vigra.defaultAxistags("xyc"))

    seg = vigra.VigraArray(labeling,axistags=vigra.defaultAxistags("xy"))

    print "done"
    seg         = vigra.analysis.labelImage(seg.astype(numpy.uint32))
    tgrid       = cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
    cgp         = cgp2d.Cgp(tgrid)



    #features    = cgp.accumulateCellFeatures(cellType=2,image=jhist.reshape([dx,dy,-1]),features="Mean")[0]['Mean']
    #features = preprocessing.scale(features)

    allF=numpy.concatenate([hist],axis=2)
    features    = cgp.accumulateCellFeatures(cellType=2,image=allF,features="Mean")[0]['Mean']


    print "features",features.shape

    imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
    imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)
    #cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)





    ###############################################################################
    # segment
    ###############################################################################


    cell1StateMixed = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)

    guessK  = int(numpy.sqrt(cgp.numCells(1)/2))
    #for kk in [2,4,8,16,32,64,128,512,cgp.numCells(2)/2]:
    for kk in [guessK]:#,4,8,16,32,64,128,512,cgp.numCells(2)/2]:

        segmentor = gseg.segmentors.HierarchicalClustering(cgp=cgp)
        # whiten the features
        features=preprocessing.scale(features)
        segmentor.segment(features,kk)
        labels 	= segmentor.labels 

        print segmentor.ward.children_


        cell1State = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)
        cell1Bounds=cgp.cell1BoundsArray()-1





        for ci  in xrange(cgp.numCells(1)):
        	
        	r1,r2  = cell1Bounds[ci,:]
        	if labels[r1]!=labels[r2]:
        		cell1State[ci]=1
        
        cell1StateMixed+=cell1State


        tgrid2 = cgp.merge2Cells(cell1State)
        cgp2   = cgp2d.Cgp(tgrid2)

        #cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))
        cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp2)#,edge_data_in=cell1State.astype(numpy.float32))
        




        #############################
        # get the local mean color
        ##############################
        print "accumulate cell mean"
        cellMean = cgp.accumulateCellFeatures(cellType=2,image=imgTopo,features="Mean")[0]['Mean']

        cells2=cgp.cells(2)
        newLabels =labels.copy()
        for i1  in range(cgp.numCells(1)):

            r1,r2 = cell1Bounds[i1,:]
            l1,l2 = labels[r1],labels[r2]

            if(l1!=l2): 

                print r1,r2



                for r in [r1,r2]:

                    # get ALL adj. regions
                    cell2 = cells2[int(r)]
                    adjCells = numpy.array([c[0] -1 for c in  cell2.adjacencyGen()])
                    #print "adj cells",adjCells
                    adjCellLabels = newLabels[adjCells]

                    if len(adjCells)<10:
                        print adjCells
                        print adjCellLabels

                        localLabels = numpy.unique(adjCellLabels)
                        print "labels",localLabels

           
                        lToDense = dict()
                        for i,l in enumerate(localLabels):
                            print "i,l",i,l
                            lToDense[l]=i


                        nLocalLabels= len(localLabels)
                        distTo = numpy.zeros(nLocalLabels)
                        coutTo = numpy.zeros(nLocalLabels)
                        ownMean = cellMean[r,:]

                        for c,l in zip(adjCells,adjCellLabels):

                            otherMean = cellMean[c,:]
                            # distance 

                            d = numpy.sum((otherMean-ownMean)**2)
                            #print d
                            distTo[lToDense[l]]+=d
                            coutTo[lToDense[l]]+=1.0

                        distTo/=coutTo

                        argmin = numpy.argsort(distTo)
                        minL = argmin[0]
                        minL = localLabels[minL]
                        newLabels[r]=minL
                        print "argumin",argmin

                    else :
                        print "\nAAAA LOOOOOOOT\n"



        cell1State = numpy.zeros(cgp.numCells(1),dtype=numpy.uint32)
        for ci  in xrange(cgp.numCells(1)):
            r1,r2  = cell1Bounds[ci,:]
            if newLabels[r1]!=newLabels[r2]:
                cell1State[ci]=1

        tgrid3 = cgp.merge2Cells(cell1State)
        cgp3   = cgp2d.Cgp(tgrid3)

        #cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1State.astype(numpy.float32))
        cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp3)#,edge_data_in=cell1State.astype(numpy.float32))
        

    print "master"




    cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=cell1StateMixed.astype(numpy.float32),cmap="hot")
    #cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp2)#,edge_data_in=cell1State.astype(numpy.float32))