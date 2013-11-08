from sklearn.cluster import MiniBatchKMeans  
import numpy
import vigra




def kMeansColoring(features,nClusters):
	segmentor = MiniBatchKMeans(n_clusters=nClusters)
	f = features.reshape(features.shape[0]*features.shape[1],-1)
	labels = segmentor.fit_predict(f)
	labels = labels.reshape([features.shape[0],features.shape[1]])
	return labels