import vigra
import numpy
import gseg

import mdp
import pylab



###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
#filepath    = 'zebra.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]

#img = pylab.imread('42049.jpg')
# transform to grayscale
im = numpy.array(numpy.sqrt((img[:,:,:3]**2.).mean(2)))


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel


matplotlib.rcParams['font.size'] = 9


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        kernelName=kernel[1]
        kernel=kernel[0]
        filtered = nd.convolve(image, kernel, mode='wrap')
        #filtered = numpy.abs(filtered)
        f = pylab.figure()
        for n, iii in enumerate([filtered,kernel]):
            #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
            f.add_subplot(1, 2, n)  # this line outputs images side-by-side
            iii-=iii.min()
            iii/=iii.max()
            pylab.imshow(numpy.swapaxes(iii,0,1),cmap='gray',interpolation="nearest")
            pylab.title(kernelName)
        pylab.show()


        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


# prepare filter bank kernels
kernels = []

for theta in range(12):
    #theta=float(theta)/2.0
    theta = theta / 12. * np.pi
    for sigma in (1.5,):
        for frequency in (0.3,):#0.085,):

            name = "theta Pi* %f sigma %f f %f"%(theta/(np.pi),sigma,frequency)

            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append([kernel,name])


compute_feats(numpy.sum(img,axis=2),kernels)