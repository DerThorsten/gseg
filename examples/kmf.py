import vigra
import numpy
import gseg

import pylab
from sklearn.feature_extraction import image as skli

from scipy.optimize import curve_fit
 
from sklearn.cluster import KMeans,MiniBatchKMeans



###############################################################################
# Generate data
###############################################################################
visu 	 	= True
filepath 	= '42049.jpg'
filepath    = '156065.jpg'
filepath    = 'zebra.jpg'
#filepath    = 't.jpg'
#filepath    = 'img.png'
rgb 		= vigra.readImage(filepath)#[0:200,0:200,:]
lab         = vigra.colors.transform_RGB2Lab(rgb)  
g           = lab[:,:,0]

k           = 3


rr= [5]
res = numpy.zeros(g.shape+(len(rr),))
resk = numpy.zeros(g.shape+(3,len(rr)))



def  kmf(X,k,initC=None):
    if initC is not None :
        km=MiniBatchKMeans(k,init=initC,max_iter=1)
    else :
        km=MiniBatchKMeans(k,max_iter=1)
    km.fit(X)
    centers = km.cluster_centers_
    return centers
    #print centers


for ir,r in enumerate(rr):

    # Disk definition:
    (center_x, center_y) = (r, r)
    radius = r

    x_grid, y_grid = numpy.meshgrid(numpy.arange(r*2+1), numpy.arange(r*2+1))
    disk = ((x_grid-center_x)**2 + (y_grid-center_y)**2) <= radius**2
    print disk.shape
    #pylab.imshow(numpy.swapaxes(disk,0,1))
    #pylab.show()
    centers  = None
    for x in range(50,lab.shape[0]-50):
        print x
        for y in range(50,lab.shape[1]-50):
            #print x,y
            patch = g[x-r:x+r+1,y-r:y+r+1]
            #print patch.shape
            dpatch = patch[disk]

            dpatch = dpatch.reshape([len(dpatch),1])
            #print dpatch.shape


            centers = kmf(dpatch,3,initC=centers)
            centersS = numpy.sort(centers.reshape([k]))


            """
            mi,ma = dpatch.min(),dpatch.max()

            ratio = (mi+1.0)/ma
            res[x,y,ir]=ratio   
            """

            resk[x,y,0,ir]=centersS[0]
            resk[x,y,1,ir]=centersS[1]
            resk[x,y,2,ir]=centersS[2]

mean = numpy.mean(res,axis=2)
mean = numpy.mean(resk,axis=3)

mean-=mean.min()
mean/=mean.max()

pylab.imshow(numpy.swapaxes(mean,0,1))
pylab.show()







rgbs = vigra.filters.nonlinearDiffusion(rgb,scale=5,edgeThreshold=1.0)


diff = numpy.sum( (rgb-lab)**2 , axis=2)
lab         = vigra.colors.transform_RGB2Lab(rgbs)  

def norm(a):
    b=a.copy()
    b-=b.min()
    b/=b.max()
    return b




import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt



def fitFunc(x,w,o,p,s):
    r = (numpy.sin(x*w+p)+0.5)
    return numpy.round(  r ,decimals=0    )*s+o











def getLine(data,x0,x1,y0,y1,n):
    #-- Extract the line...
    # Make a line with "num" points...
    x0, y0 = 0, rgb.shape[1]/2-30 # These are in _pixel_ coordinates!!
    x1, y1 = rgb.shape[0]-1 , rgb.shape[1]/2-30
    x, y = np.linspace(x0, x1, n), np.linspace(y0, y1, n)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(data, np.vstack((x,y)))
    return zi


#-- Extract the line...
# Make a line with "num" points...
s=-20
x0, y0 = 0, rgb.shape[1]/2+s # These are in _pixel_ coordinates!!
x1, y1 = rgb.shape[0]-1 , rgb.shape[1]/2+s
n = rgb.shape[0]
zi = getLine(data=lab[:,:,0],x0=x0,x1=x1,y0=y0,y1=y1,n=n)



size=25 

for x in range(0,n,size):

    xmin = max(0,x-size)
    xmax = min(n,x+size)
    data  = zi[xmin:xmax]

    t = numpy.arange(len(data))


    s = data.max() - data.min()
    o = data.min()
    w = 2
    p=0

    fitParams, fitCovariances = curve_fit(fitFunc, t, data,p0=[w,o,p,s])
    res=[]

    for x in t:
        res.append(fitFunc(x,*fitParams)) 
    res = numpy.array(res)
    diff  = numpy.abs(res-data)   
    #-- Plot...
    fig, axes = plt.subplots(nrows=3)
    axes[0].plot(data)
    axes[1].plot(res)
    axes[2].plot(diff)
    plt.show()




res=[]

for x in t:
    res.append(fitFunc(x,*fitParams))


#-- Plot...
fig, axes = plt.subplots(nrows=3)
axes[0].imshow(numpy.swapaxes(norm(rgb),0,1))
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(zi)
axes[2].plot(res)
plt.show()









f = pylab.figure()
for n, img in enumerate([rgbs,rgb,diff]):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 3, n)  # this line outputs images side-by-side
    pylab.imshow(norm(numpy.swapaxes(img,0,1)))
pylab.show()