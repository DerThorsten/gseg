import vigra
import numpy
import phist
import vlfeat
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def colorSpaceDescriptor(imgRgb):
	rgb = imgRgb
	lab  = vigra.colors.transform_RGB2Lab(rgb)	
	luv  = vigra.colors.transform_RGB2Luv(rgb)
	xyz  = vigra.colors.transform_RGB2XYZ(rgb)
	rgbp = vigra.colors.transform_RGB2RGBPrime(rgb)
	cbcr = vigra.colors.transform_RGBPrime2YPrimeCbCr(rgbp)
	return  numpy.concatenate([rgb,lab,luv,xyz,rgbp,cbcr],axis=2)




def gaborKernelBank(nRot=8,sigmas=[1.5]):
	pass








def denseSift(imgIn,verbose=False,border=[10,10],visu=False,dtype=numpy.float32):
    def  extendBorder(imgIn,borderSize,visu=False):
        """
            extend the image with mirror boundary conditions
        """
        img=imgIn
        oldShape = img.shape
        flipUd   = numpy.fliplr(img)
        imgUd_U  = flipUd[:,oldShape[1]-borderSize[1]:oldShape[1]]
        imgUd_D  = flipUd[:, 0: borderSize[1]]
        assert imgUd_U.shape[1]==borderSize[1]
        assert imgUd_D.shape[1]==borderSize[1]

        img = numpy.concatenate([imgUd_U,img,imgUd_D],axis=1)

        oldShape = img.shape
        flipLr   = numpy.flipud(img)
        imgLr_R  = flipLr[oldShape[0]-borderSize[0]:oldShape[0],:]
        imgLr_L  = flipLr[0: borderSize[0],:]
        assert imgLr_R.shape[0]==borderSize[0]
        assert imgLr_L.shape[0]==borderSize[0]

        img = numpy.concatenate([imgLr_R,img,imgLr_L],axis=0)

        if visu:
            plt.imshow(img.T, cmap = cm.Greys_r)
            plt.show()
        
        return img
    def frameToCoordinate(F):
        """swap axis from deseSift output and apply floor
        """
        cX = F[1,:]
        cY = F[0,:]
        cX = numpy.floor(cX)
        cY = numpy.floor(cY)
        cX = cX.astype(numpy.int32)
        cY = cY.astype(numpy.int32)
        coords = (cX,cY)
        return coords

    def checkCovering(shape,border,coord):
        """ check if the orginal image is densely covered
        """
        covering = numpy.zeros(shape)
        covering[coord] = 1

        coveringSub  = covering[border[0]:shape[0]-border[0] ,border[1]:shape[1]-border[1] ]
        whereZero    = numpy.where(coveringSub==0)

        total  = coveringSub.shape[0]*coveringSub.shape[1]
        nZeros = len(whereZero[0])
        nOnes  = total-nZeros
        return nZeros==0

    if imgIn.ndim == 3 :
        imgIn = numpy.sum(imgIn,axis=2)

    srcShape = imgIn.shape
    # extend image with mirror boundary condions
    img      = extendBorder(imgIn,border)
    img      = numpy.require(img,dtype=numpy.float32)
    # shape is bigger than src shape since the image has 
    # been extented by "extendBorder" 
    shape    = img.shape 
    
    # call *EXTERNAL* code for dense sift
    F,D=siftRes=vlfeat.vl_dsift(
        img, 
        step=-1, 
        bounds=numpy.zeros(1, 'f'), 
        size=-1, 
        fast=True, 
        verbose=verbose, 
        norm=False
    )
    # number of desc (128 default)
    numDescriptors  = D.shape[0]
    featureImg      = numpy.ones([shape[0],shape[1],numDescriptors],dtype=dtype)
    coords          = frameToCoordinate(F)

    # check that each pixel is covered
    assert checkCovering(shape,border,coords)

    # MAKE ME FAST!!!!
    # write results in one dense array
    for fi in xrange(numDescriptors):
        if verbose : print fi,",",numDescriptors,"(make me faster!!)"
        #featureImg_fi = featureImg[:,:,fi]
        featureImg[coords[0],coords[1],fi] = D[fi,:]

    # UN-extend image
    featureImgSrcShape  = featureImg[border[0]:border[0]+srcShape[0],border[1]:border[1]+srcShape[1],:].copy()    
    assert featureImgSrcShape.shape[0] == srcShape[0]
    assert featureImgSrcShape.shape[1] == srcShape[1]
    assert featureImgSrcShape.shape[2] == numDescriptors

    # visualize results ?
    if visu:
        for fi in xrange(numDescriptors):
            print fi
            plt.imshow(featureImgSrcShape[:,:,fi].T, cmap = cm.Greys_r)
            plt.show()

    return featureImgSrcShape

