import numpy
import opengm
import vigra
import cgp2d
import layerviewer as lv
from pyqtgraph.Qt import QtGui, QtCore

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
     FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer



















def sampleStateFromProbability(stateMean,stateStd):

    nVar  = len(stateMean)

    # probablity of beeing on  to maximize
    edgeProbability = numpy.ones(nvar)

    # energy of beeing on 
    edgeEnergy 
    





def optimizer(cgp,eGlobal,oracle,initStd=0.5,damping=0.5,img=None):

    nVar          = cgp.numCells(1)
    nReg          = cgp.numCells(2)
    nSamples      = 20
    nEliteSamples = 1
    maxIterations = 2000

    
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

        print "get samples (by multicut)"
        # get samples states from probablity desnisty
        samplesPrimal,samplesDual = oracle.getSamples(nSamples,outPrimal=samplesPrimal,outDual=samplesDual)

        #print "eval samples"
        widgets = ['evaluate %d Samples: '%nSamples, Percentage(), ' ', Bar(marker=RotatingMarker()),
                    ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=nSamples-1).start()

        for n in range(nSamples):
            pbar.update(n)
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

        print "iteration",iteration
        if (iteration+1)%100==0:
            
            cgp2d.visualize(img_rgb=img,cgp=cgp,edge_data_in=bestState.astype(numpy.float32))
        print "next iter"


    return bestState,bestValue



