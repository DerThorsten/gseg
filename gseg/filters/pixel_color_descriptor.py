



def colorSpaceTransform(img,fromCs,toCs):
	if fromCp == toCp :
		return img
	fname 		= "transform_%s2%s"%(fromCp,toCp)
	transformer = getattr(vigra.colors, fname)
	return transformer(img)



def toSomeColorSpaces(img,fromCs,css=['RGB','Lab','Luv','XYZ','RGBPrime','YPrimeCbCr']):
	colorspaces=dict()
	for cp in css:
		colorspaces[cp]=colorSpaceTransform)(img,fromCp=fromCp,toCp=)
	return colorspaces




