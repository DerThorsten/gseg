import numpy
import opengm
import vigra

import layerviewer as lv
from pyqtgraph.Qt import QtGui, QtCore

def sampleStateFromProbability(stateMean,stateStd):

    nVar  = len(stateMean)

    # probablity of beeing on  to maximize
    edgeProbability = numpy.ones(nvar)

    # energy of beeing on 
    edgeEnergy 
    





def optimizer(cgp,eGlobal,oracle,initStd=0.5,damping=0.9,img=None):

    nVar          = cgp.numCells(1)
    nReg          = cgp.numCells(2)
    nSamples      = 100
    nEliteSamples = 10
    maxIterations = 200

    
    stateMean =  numpy.zeros(nVar)+0.5
    stateStd  =  numpy.zeros(nVar)+initStd

    bestState = numpy.zeros(nVar,dtype=numpy.uint32)
    bestValue = float('inf')

    bestSeg   = numpy.zeros(nReg,dtype=numpy.uint32)

    samplesPrimal = numpy.zeros( [nSamples,nReg]  ,dtype=opengm.label_type)
    samplesDual   = numpy.zeros( [nSamples,nVar]  ,dtype=opengm.label_type)

    sampleEnergy = numpy.zeros(nSamples)

    for iteration in range(maxIterations):

        #print "update probability"
        # update segmentation oracles probability
        oracle.updateDistribution(stateMean,stateStd)

        #print "get samples (by multicut)"
        # get samples states from probablity desnisty
        samplesPrimal,samplesDual = oracle.getSamples(nSamples,outPrimal=samplesPrimal,outDual=samplesDual)

        #print "eval samples"
        # evaluate samples
        for n in range(nSamples):
            #print n
            sampleEnergy[n]= eGlobal(argPrimal=samplesPrimal[n,:],argDual=samplesDual[n,:])

        print "value ",bestValue


        #print "sort samples"
        # sort samples by energy
        sortedIndex  = numpy.argsort(sampleEnergy)

        #print "get elite samples"
        # get elitem samples
        eliteSamplesDual    = samplesDual[ sortedIndex[0:nEliteSamples] ,:]
        eliteSamplesPrimal  = samplesPrimal[ sortedIndex[0:nEliteSamples] ,:]
        if(sampleEnergy[sortedIndex[0]]<bestValue):
            bestState[:] = eliteSamplesDual[0,:]
            bestSeg[:]   = eliteSamplesPrimal[0,:]
            bestValue    = sampleEnergy[sortedIndex[0]]

        #print "shift samples"
        # shift probability density 
        #print "elite sahpe",eliteSamplesDual.shape

        stateMean = stateMean*damping + (1.0-damping)*numpy.mean(eliteSamplesDual,axis=0)
        stateStd  = stateStd*damping  + (1.0-damping)*numpy.std(eliteSamplesDual,axis=0)

        #print "visu"
        if (iteration+1)%40==0:
            
            app = QtGui.QApplication([])
            viewer =  lv.LayerViewer()
            #print "add layer"
            viewer.addLayer(name='img',layerType='GrayLayer')

            #print "add layer"
            viewer.addLayer(name='denseImage',layerType='SegmentationLayer')


           
            viewer.show()

            # input gray layer
            #viewer.addLayer(name='LabelImage',layerType='GrayLayer')
            #viewer.setLayerData(name='LabelImage',data=labelImage)
            #print "imgshape",img.shape

            # print "get label image"
            # get feature image
            labelImage = cgp.featureToImage(
                cellType=2,
                features=bestSeg.astype(numpy.float32),
                ignoreInactive=True,
                useTopologicalShape=False
            )
            denseLabels = vigra.analysis.labelImage(labelImage)
            print "add layer"
            viewer.setLayerData(name='img',data=img)

            print "add layer"
            viewer.setLayerData(name='denseImage',data=denseLabels)


            viewer.autoRange()
            QtGui.QApplication.instance().exec_()
        print "next iter"


    return bestState,bestValue



