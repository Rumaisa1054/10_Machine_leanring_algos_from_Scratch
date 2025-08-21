import matplotlib.pyplot as plt
import numpy as np
class kMeans:
    def __init__(self,K=5,max_iter=500,plot = False):
        self.K = K
        self.max_iter = max_iter
        self.plott = plot
        self.clusters = [[] for _ in range(self.K)]
        self.centeriod = []
    def eucl(self,sample, centeriod):
        return np.sqrt(np.sum((sample - centeriod)**2))
    def closest_centroid(self,sample,centeriods):
        # find the new centeriods
        distances = [self.eucl(sample, c) for c in centeriods]
        closest_ind = np.argmin(distances)
        return closest_ind
    def create_cluster(self,centeriod):
        # assign cluster to each sample
        clusters = [[] for _ in range(self.K)]
        for ind, sample in enumerate(self.X):
            i = self.closest_centroid(sample,centeriod)
            clusters[i].append(ind)
        return  clusters
    def new_centriod(self,clusters):
        # assign mean value of clusters to centroid
        # creating list of list
        cent = np.zeros((self.K,self.n_features))
        for ind,clus in enumerate(clusters):
            clus_mean = np.mean(self.X[clus],axis=0)
            cent[ind] = clus_mean
        return cent

    def are_same_(self,old_centriod,new_ones):
        # if old and new centers are same
        d = [self.eucl(old_centriod,new_ones) for i in range(self.K)]
        return np.sum(d) == 0
    def plot(self):
        fig,ax = plt.subplots(figsize=(12,8))
        for row,element_ind in enumerate(self.clusters):
            point = self.X[element_ind].T
            ax.scatter(*point)
        for point in self.centeriod:
            ax.scatter(*point,marker = "*",color = "red",linewidth = 2)
        plt.show()

    def get_cluster_label(self,clusters):
        #once the clusters are identified _ just get their label
        P = np.empty(self.n_Samples)
        # clusters is a list of list - rows = cluster and inside elemente are indexes of samples in x
        for row, cluster in enumerate(clusters):
            for sample_index in cluster:
                P[sample_index] = row
        return P

    def predict(self,X):
        self.X = X
        self.n_features = X.shape[1]
        self.n_Samples = X.shape[0]
        # select K random indexes (starting  from 0 to no of samples/rows in X)
        random_samples_indexes = np.random.choice(self.n_Samples,size=self.K)
        self.centeriod = [X[index] for index in random_samples_indexes]
        self.centeriod = np.array(self.centeriod)

        for iteration in range(self.max_iter):
            self.clusters = self.create_cluster(self.centeriod)
            if self.plott: self.plot()
            old_centriod = self.centeriod
            self.centeriod= self.new_centriod(self.clusters)
            if self.are_same_(self.centeriod,old_centriod):
                break
        return self.get_cluster_label(self.clusters)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=1.05, random_state=42)

    kmeans = kMeans(K=3, max_iter=100, plot=True)
    y_pred = kmeans.predict(X)
