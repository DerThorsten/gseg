import vigra
import numpy
import gseg
import phist
import pylab
from sklearn.decomposition import PCA, KernelPCA,TruncatedSVD,NMF,ProjectedGradientNMF,RandomizedPCA,SparsePCA
from sklearn import preprocessing


###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
#filepath    = 'zebra.jpg'
#filepath    = 't.jpg'
filepath    = 'img.png'
rgb 		= vigra.readImage(filepath)#[0:200,0:200,:]
lab         = vigra.colors.transform_RGB2Lab(rgb)  




def deepHistogram(image,dims=3,lowBins=10):
    features  = image
    dx,dy     = features.shape[0],features.shape[1]
    nPixel    = dx*dy
    featuresINIT  = image.copy()
    featuresINIT  = featuresINIT.reshape([dx,dy,-1])
    featuresINIT  = featuresINIT.reshape([nPixel,-1])
    
    print "init hist"
  
    while(True):
        features  = phist.histogram(features,r=1,bins=20,sigma=[2.0,1.0])
        features  = features.reshape([dx,dy,-1])
        features  = features.reshape([nPixel,-1])
        #take features and reduce dim
        reducer = PCA(n_components=dims)
        print "reduce dim pca"
        fLow    = reducer.fit_transform(features).reshape([dx,dy,dims]).copy()
        fLow    = fLow.astype(numpy.float32)
        fLow-=fLow.min()
        fLow/=fLow.max()
        f = pylab.figure()
        for n, img in enumerate([fLow[:,:,0:3]]):
            imgs = img.copy()
            imgs-=imgs.min()
            imgs/=imgs.max()

            #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
            f.add_subplot(1, 1, n)  # this line outputs images side-by-side
            pylab.imshow(numpy.swapaxes(imgs,0,1))
        pylab.show()


        features  = features.reshape([dx,dy,-1])
        features  = features.reshape([nPixel,-1])

        # mix in 
        a = preprocessing.scale(featuresINIT)*0.1
        b = preprocessing.scale(features)

        features = numpy.concatenate([a,b],axis=1)
        features-=features.min()
        features+=0.001
        features/=features.max()
        features  = features.reshape([dx,dy,-1])
f=numpy.concatenate([lab[:,:,1:3],lab],axis=2)
f=vigra.taggedView(f,axistags=vigra.defaultAxistags("xyc"))
deepHistogram(image=f)








f = pylab.figure()
for n, img in enumerate([rgbs,rgb,diff]):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 3, n)  # this line outputs images side-by-side
    pylab.imshow(norm(numpy.swapaxes(img,0,1)))
pylab.show()