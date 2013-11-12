import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt
import pylab

import gseg



def flip(img):
	img=numpy.squeeze(img)
	return numpy.swapaxes(img,0,1)






visu 	 	= True
#filepath 	= '42049.jpg'
filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
gradmag  	= vigra.filters.gaussianGradientMagnitude(img,1.0)

gradmag=numpy.squeeze(gradmag)
gradmag-=gradmag.min()
gradmag/=gradmag.max()







def patchNormalization(img,patchR=50,overlap=10):
	output = numpy.zeros(img.shape)
	shape=img.shape


	for x in range(0,shape[0],overlap):
		for y in range(0,shape[1],overlap):
			print x,y


			patch      =  img[  max(x-patchR,0):min(x+patchR+1,shape[0]) , max(y-patchR,0):min(y+patchR+1,shape[1])   ]
			patchOut   =  output[  max(x-patchR,0):min(x+patchR+1,shape[0]) , max(y-patchR,0):min(y+patchR+1,shape[1])   ]
			data = patch.copy()
			data-=data.min()
			data/=data.max()

			patchOut+=data

			print patch.shape

	output-=output.min()
	output/=output.max()

	return output

pgradmag = patchNormalization(gradmag)




f = pylab.figure()
for n, img in enumerate([pgradmag,gradmag]):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(img,0,1))
pylab.show()



plt.imshow(flip(gradmag))
plt.show()
