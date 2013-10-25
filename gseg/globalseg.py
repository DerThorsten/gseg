


"""
X  are region  variables


INPUT:
	E_G(X) :
		- is a very high order function which should be minimized

	DIFFERENT_REDUCED_SEG  = { watershed, mc on gradient , etc }

WORKING DATA:
	P_L(X) :
		- is the probability desity  which will be modified 

		- it can be a graphical model of any structure

		- the structure of the graphical model
		  should resamble somehow E_G(X)



ALGORITHM:
	
Initialization : 
	- evaluate all seg in DIFFERENT_REDUCED_SEG

	- get elite samples from evaluation

	- update probability from elite samples

Iteration :
	


"""


# defined by user

def globalEnergyFunction(state):
    pass


def sampleStateFromProbability(stateMean,stateStd):

    nVar  = len(stateMean)

    # probablity of beeing on  to maximize
    edgeProbability = numpy.ones(nvar)

    # energy of beeing on 
    edgeEnergy 
    


def probabilityToEnergy(p,out=None):
    pass

def sampleFromGauss(mean,std,out=None):
    pass


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



"""
    - multicut  (|V|-Class with permutable labels )
    - multi-instancce (V|-Class , unique cells vs bg, but different cells might touch,semi permutable labels)
    - supervised ( N-Class Seg )

"""

def optimizer(cgp,initStd=0.5,damping=0.5,globalEnergyFunction,oracle):


    nVar          = cgp.numCells(2)
    nSamples      = 1000
    nEliteSamples = 20 
    maxIterations = 100

    
    stateMean =  numpy.zeros(nVar)+0.5
    stateStd  =  numpy.zeros(nVar)+initStd

    bestState = numpy.zeros(nVar,dtype=uint32)
    bestValue = float('inf')

    for iteration in range(maxIterations):

        # update segmentation oracles probability
        oracle.updateDistribution(stateMean,stateStd)

        # pass warm start (test for speedup)
        oracle.setWarmStartCandidates([bestState])

        # get samples states from probablity desnisty
        samplesPrimal,samplesDual = oracle.getSamples(nSamples)

        # evaluate samples
        sampleEnergy = numpy.array([ globalEnergyFunction(s) for sp,sd in zip(samplesPrimal,samplesDual)])

        








        # sort samples by energy
        sortedIndex  = numpy.argsort(sampleEnergy)

        # get elitem samples
        eliteSamples  = numpy.array(sampleStates[sortedIndex[0:nEliteSamples]])

        if(sampleEnergy[sortedIndex[0]]<bestValue):
            bestState[:] = eliteSamples[0][:]
            bestValue    = sampleEnergy[sortedIndex[0]]


        # shift probability density 
        stateMean = stateMean*damping + (1.0-damping)*numpy.mean(eliteSamples,axis=1)
        stateStd  = stateStd*damping  + (1.0-damping)*numpy.std(eliteSamples,axis=1)

    return bestState,bestValue



