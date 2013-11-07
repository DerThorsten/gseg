import numpy
import opengm
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
     FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer



def sampleFromGauss(mean,std,out,clip=[0.001,0.999]):
    #print "mean",mean.shape
    #print "std",std.shape
    #print "out",out.shape

    assert len(mean)==len(std)
    assert len(out)==len(mean)

    n = len(mean)

    samples  = numpy.random.standard_normal(n)
    samples *=std
    samples +=mean



    
    samples =  numpy.clip(samples,clip[0],clip[1])
    #print samples[0:10]
    return samples

def probabilityToWeights(p1,out,beta=0.5):
    assert len(out)==len(p1)
    p0 = 1.0 - p1
    out[:]=numpy.log( p0 / p1 ) + numpy.log((1.0-beta)/beta)
    return out



class MulticutOracle(object):
    def __init__(self,cgp,beta=0.5):
        self.cgp=cgp
        self.probability =  numpy.ones(cgp.numCells(1),dtype=numpy.float64)
        self.weights     =  numpy.ones(cgp.numCells(1),dtype=numpy.float64)
        self.mean        =  numpy.ones(cgp.numCells(1),dtype=numpy.float64)
        self.std         =  numpy.ones(cgp.numCells(1),dtype=numpy.float64)

        # generate graphical model 
        self.cgc        = None
        self.beta=beta

        boundArray = self.cgp.cell1BoundsArray()-1

        nVar = cgp.numCells(2)
        nFac = cgp.numCells(1)
        space = numpy.ones(nVar,dtype=opengm.label_type)*nVar
        self.gm = opengm.gm(space)

        wZero  = numpy.zeros(nFac,dtype=opengm.value_type)

        pf=opengm.pottsFunctions([nVar,nVar],wZero,wZero)

        fids = self.gm.addFunctions(pf)
        self.gm.addFactors(fids,boundArray)
        self.cgc = opengm.inference.Cgc(gm=self.gm,parameter=opengm.InfParam(planar=True))




    def updateDistribution(self,mean,std):
        self.mean = mean
        self.std  = std




    def getSamples(self,n,outPrimal,outDual):
        widgets = ['get %d Multicut Samples: '%n, Percentage(), ' ', Bar(marker=RotatingMarker()),
                    ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=n).start()

        #widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
        #pbar = ProgressBar(widgets=widgets, maxval=n).start()



        for s in range(n):
            pbar.update(s)

            # get a sampled probability
            self.probability=sampleFromGauss(mean=self.mean,std=self.std,out=self.probability)

            # convert to energz
            self.weights=probabilityToWeights(p1=self.probability,out=self.weights,beta=self.beta)

            # update multicut weights
            self.cgc.changeWeights(self.weights)

            # infer
            #self.cgc.infer(self.cgc.verboseVisitor())
            self.cgc.infer()

            # store result
            outDual[s,:]    = self.cgc.argDual(out=outDual[s,:])
            outPrimal[s,:]  = self.cgc.arg()

        pbar.finish()


        return outPrimal,outDual

