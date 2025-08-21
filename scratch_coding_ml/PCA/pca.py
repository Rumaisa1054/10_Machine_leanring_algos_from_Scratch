import numpy as np

class PCA:
    def __init__(self,n_comp):
        self.n_components = n_comp
        self.components = None
        self.mean = None
    def fit(self,X):
        # hr column ka mean nikal lo - mean centering X_i - x_mean
        self.mean = np.mean(X,axis=0)
        X = X - self.mean

        # cov- x needs to be transposed because this function needs the
        # samples as columns
        cov_mat = np.cov(X.T)
        # store them as val,vec pair
        vals,vec = np.linalg.eig(cov_mat)
        # sort
        # sort eigenvalues & eigenvectors
        idxs = np.argsort(vals)[::-1]
        eigvecs = vec[:, idxs]

        # store top n_components
        self.components = eigvecs[:, :self.n_components]

    def transform(self,X):
        X = X - self.mean
        return np.dot(X,self.components)

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    X, y = load_digits(return_X_y=True)

    # baseline (no PCA)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train1, y_train1)
    predictions1 = knn.predict(X_test1)

    # with PCA
    pca = PCA(n_comp=len(np.unique(y)) // 2)  # reduce dimension
    pca.fit(X)
    X_reduced = pca.transform(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("Accuracy with PCA:    ", accuracy_score(y_test, predictions))
    print("Accuracy without PCA: ", accuracy_score(y_test1, predictions1))
