import vigra
import opengm
import cgp2d

import numpy
import matplotlib.pyplot as plt


import gseg









visu 	 	= True
filepath 	= '42049.jpg'
#filepath    = '156065.jpg'
img 		= vigra.readImage(filepath)#[0:200,0:200,:]
imgLab  	= vigra.colors.transform_RGB2Lab(img)
gradmag  	= vigra.filters.gaussianGradientMagnitude(img,4.0)

seg,nseg    = vigra.analysis.slicSuperpixels(imgLab,10.0,5)
#seg,nseg 	= vigra.analysis.watersheds(gradmag)

tgrid 	= cgp2d.TopologicalGrid(seg.astype(numpy.uint64))
cgp  	= cgp2d.Cgp(tgrid)


imgTopo  	= vigra.sampling.resize(imgLab,cgp.shape)
imgRGBTopo  = vigra.colors.transform_Lab2RGB(imgTopo)
gradTopo 	= vigra.filters.gaussianGradientMagnitude(imgTopo,1.0)
labelsTopo  = vigra.sampling.resize(seg.astype(numpy.float32),cgp.shape,0)



nVar 	= cgp.numCells(2)
nFac 	= cgp.numCells(1)
space 	= numpy.ones(nVar,dtype=opengm.label_type)*nVar
gm   	= opengm.gm(space)
wZero  	= numpy.zeros(nFac,dtype=opengm.value_type)
pf 		= opengm.pottsFunctions([nVar,nVar],wZero,wZero)
fids 	= gm.addFunctions(pf)
gm.addFactors(fids,cgp.cell1BoundsArray()-1)
cgc 	= opengm.inference.Cgc(gm=gm,parameter=opengm.InfParam(planar=True))


# visualize segmetation

#cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp)#,edge_data_in=bestState.astype(numpy.float32))


argDual  = numpy.zeros(cgp.numCells(1),dtype=numpy.uint64)


sigmas 		= [1.0 , 2.0, 4.0 ,6.0]
features	= numpy.zeros([len(sigmas),cgp.numCells(1)],dtype=numpy.float32)

for i,s in enumerate(sigmas):
	gradMag = vigra.filters.gaussianGradientMagnitude(imgTopo,s)
	f,a=cgp.accumulateCellFeatures(cellType=1,image=gradMag,features="Mean")
	#print f

	features[i,:]=f['Mean']


g 	  = numpy.sum(features,axis=0)


for gamma in [0.6,0.3,0.5,0.7,0.9,1.1,1.2]:




	e1  = numpy.exp(-1.0*gamma*g)
	e0  = 1.0 - e1
	w   = e1 - e0 

	print "gamma ",gamma
	print "grad",g[0:5]
	print "e0  "  ,e0[0:5]
	print "e1  "  ,e1[0:5]
	print "w   "  ,w[0:5]

	cgc.changeWeights(w)
	cgc.infer(cgc.verboseVisitor())
	#cgc.infer()
	argDual = cgc.argDual(out=argDual)

	cgp2d.visualize(img_rgb=imgRGBTopo,cgp=cgp,edge_data_in=argDual.astype(numpy.float32))







