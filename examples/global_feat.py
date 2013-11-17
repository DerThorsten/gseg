import vigra
import opengm
import cgp2d
import numpy
import gseg
import phist



import cv2.cv as cv
import cv2
 
def build_filters():
	""" returns a list of kernels in several orientations
	"""
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 32):
		params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
		'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
		kern = cv2.getGaborKernel(**params)
		kern /= 1.5*kern.sum()
		filters.append((kern,params))
	return filters
 
def process(img, filters):
	""" returns the img filtered by the filter list
	"""
	accum = np.zeros_like(img)
	for kern,params in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum









###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'

img 		= vigra.readImage(filepath)#[0:200,0:200,:]


def norm(a):
	b=a.copy()
	b-=b.min()
	b/=b.max()
	return b
def globalFeatures(img,s=10,sf=15.0):

	
	imgLab  	= vigra.colors.transform_RGB2Lab(img)
	seg,nseg    = vigra.analysis.slicSuperpixels(imgLab,sf,s)
	seg 		= vigra.analysis.labelImage(seg)
	tgrid 		= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
	cgp  		= cgp2d.Cgp(tgrid)

	def toCgpShape(imgLocal):
		return vigra.sampling.resize(imgLocal,cgp.shape)
	imgTopo  	= toCgpShape(imgLab)
	imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)




	gradMag 	= numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgLab,sigma=1.0))

	# mean color accumulation
	regionLab = cgp.accumulateCellFeatures(cellType=2,image=imgLab,features="Mean")[0]['Mean']

	# 
	gradSum = cgp.accumulateCellFeatures(cellType=1,image=gradMag,features="Sum")[0]['Sum'].astype(numpy.float32)

	print gradSum.shape,gradSum.dtype

	cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=norm(gradSum))

	print "sum",numpy.sum(gradSum)/imgLab.size




globalFeatures(img)