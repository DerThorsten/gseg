import numpy
import vigra






class FeatureFunctions(object):

    class GaussianGradientMagnitude(object):

        def __init__(self,img,sigmaRange=[0.5,5.0,0.2] ):

            self._sigmas    = [s for s in sigmaRange]
            self._features  = numpy.zeros([len(self._sigmas),img.shape[0],img.shape[1]])
            # compute the gradients 
            for i,sigma in enumerate(self._sigmas):
                self._features[i,:,:]=vigra.filters.GaussianGradientMagnitude(img,sigma)
