import vigra
import numpy
import phist



def colorSpaceDescriptor(imgRgb):
	rgb = imgRgb
	lab  = vigra.colors.transform_RGB2Lab(rgb)	
	luv  = vigra.colors.transform_RGB2Luv(rgb)
	xyz  = vigra.colors.transform_RGB2XYZ(rgb)
	rgbp = vigra.colors.transform_RGB2RGBPrime(rgb)
	cbcr = vigra.colors.transform_RGBPrime2YPrimeCbCr(rgbp)
	return  numpy.concatenate([rgb,lab,luv,xyz,rgbp,cbcr],axis=2)




def jointColorHistDescriptor(img,r=1,bins=5,sigma=[1.0,1.0]):
	nCp 	 = img.shape[2]/3
	outShape = [img.shape[0],img.shape[1],nCp,bins,bins,bins]

	print "allocate ",outShape
	out      = numpy.zeros(outShape,dtype=numpy.float32)


	for cp in range(nCp): 
		inputImg = img[:,:,cp*3:(cp+1)*3]
		cOut = out[:,:,cp,:,:,:]
		cOut = phist.jointHistogram(image=inputImg,bins=bins,r=r,sigma=sigma,out=cOut)


	return out


