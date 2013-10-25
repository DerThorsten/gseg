class MulticutOracle(object):
    def __init__(self,cgp):
        self.cgp=cgp
        self.probablity = numpy.ones(cgp.numCells(1),dtype=numpy.float64)
        self.energy     = numpy.ones(cgp.numCells(1),dtype=numpy.float64)
        self.mean       = None
        self.std        = None

        # generate graphical model 
        self.gm         = None
        self.cgc        = None


    def updateDistribution(self,mean,std):
        self.mean = mean
        self.std  = std




    def getSamples(self,n,outPrimal=None,outDual=None):
        
        for s in range(n):

            # get a sampled probability
            self.probability=sampleFromGauss(mean=self.mean,std=self.std,out=self.probability)

            # convert to energz
            self.energy=probabilityToEnergy(p=self.probability,out=self.energy)

            # update multicut weights
            self.cgc.updateWeighs(self.energy)

            # infer
            self.cgc.infer()

            # store result
            outDual[s,:]    = self.cgc.argDual()
            outPrimal[s,:]  = self.cgc.arg()

        return outPrimal,outDual

