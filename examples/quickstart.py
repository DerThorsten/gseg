import vigra
import opengm
import numpy
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import layerviewer as lv


import gseg


filepath = 'cell.jpg'
img		 = numpy.squeeze(vigra.readImage(filepath))
gradmag  = vigra.filters.gaussianGradientMagnitude(img,2.0)

seg,nseg = vigra.analysis.watersheds(gradmag)




app = QtGui.QApplication([])
viewer =  lv.LayerViewer()
viewer.show()



# input gray layer
viewer.addLayer(name='Input',layerType='GrayLayer')
viewer.setLayerData(name='Input',data=img)

# gradmag gray layer
viewer.addLayer(name='GradMag',layerType='GrayLayer')
viewer.setLayerData(name='GradMag',data=gradmag)

# seg layer
viewer.addLayer(name='SuperPixels',layerType='SegmentationLayer')
viewer.setLayerData(name='SuperPixels',data=seg)




# segmentation layer



viewer.autoRange()


QtGui.QApplication.instance().exec_()


