
import numpy





# label histogram bin similarity 


centers = numpy.array(
    [ 
        [0,       10,     255 ],
        [0,       20,     255 ],
        [10,      10,     10  ],
        [30,      30,     30  ],
        [100,     100,    100 ] 
    ],dtype=numpy.float32
)   


def binsim(centers,norm,gamma):

    c = centers.copy()
    c-=c.min()
    c/=c.max()
    k = centers.shape[0]
    f = centers.shape[1]

    print "k",k,"f",f

    diffarray  = numpy.zeros([k,k],dtype=numpy.float32)



    for k1 in range(k-1):
        for k2 in range(k1+1,k):
            d = numpy.sum(numpy.abs(centers[k1,:]-centers[k2,:])**norm)
            #print k1,k2,"diffssss",d
            diffarray[k1,k2]=d
            diffarray[k2,k1]=d


    r = numpy.exp(-gamma*diffarray)


    for kk in range(k):
        diffarray[kk,kk]=1.0

    for kk in range(k):
        r[kk,:]=r[kk,:]/numpy.sum(r[kk,:])
    print r

    for k1 in range(k-1):
        print k1,k1,"diffssss",r[k1,k1]
        for k2 in range(k1+1,k):
            d = numpy.sum(numpy.abs(centers[k1,:]-centers[k2,:])**norm)
            print k1,k2,"diffssss",r[k1,k2]
binsim(centers,norm=1,gamma=0.05)