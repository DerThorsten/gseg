import vigra
import numpy
import phist
import skimage.color
import pylab


def show(img):
	i=img.copy()
	i-=i.min()
	i/=i.max()
	pylab.imshow(numpy.swapaxes(i,0,1))
	pylab.show()

	for c in range(3):
		print c,
		cc=img[:,:,c].copy()
		cc-=cc.min()
		cc/=cc.max()
		pylab.imshow(numpy.swapaxes(cc,0,1))
		pylab.show()


def colorSpaceDescriptor(imgRgb):
	rgb = imgRgb
	rgbINTEGRAL = rgb.astype(numpy.uint8)

	lab  = vigra.colors.transform_RGB2Lab(rgb)	
	luv  = vigra.colors.transform_RGB2Luv(rgb)
	xyz  = vigra.colors.transform_RGB2XYZ(rgb)
	rgbp = vigra.colors.transform_RGB2RGBPrime(rgb)
	cbcr = vigra.colors.transform_RGBPrime2YPrimeCbCr(rgbp)
	hed  = skimage.color.rgb2hed(numpy.array(rgbINTEGRAL))
	hsv  = skimage.color.rgb2hsv(numpy.array(rgbINTEGRAL))


	rgb2  = skimage.color.hed2rgb(numpy.array(hed))

	print "HED"
	show(hed)
	print "HSV"
	show(hsv)
	return  numpy.concatenate([rgb,lab,luv,xyz,rgbp,cbcr],axis=2)






