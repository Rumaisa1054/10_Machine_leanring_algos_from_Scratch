import numpy as np
from collections import defaultdict
class NaiveBayes:
    def __init__(self):
        pass
    def fit(self,X,y):
        samples, no_features = X.shape
        self.classes = np.unique(y)
        no_classes = len(self.classes)

        # step : 01 calculate mean - sd - prior (freq of each class)
        # for mean of each feature
        self._mean = np.zeros((no_classes, no_features), dtype=np.float32)
        # for std of each feature
        self._var = np.zeros((no_classes, no_features), dtype=np.float32)
        self.prior = np.zeros(no_classes,dtype=np.float32) # p(y)

        for index,clas in enumerate(self.classes):
            # only get the samples which belong to this class
            X_c = X[y == clas]
            self._mean[index,:] = X_c.mean(axis=0) # column wise
            self._var[index,:] = X_c.var(axis=0) # column wise
            self.prior[index] = X_c.shape[0] / samples


    def _pdf(self,index,x):
        mean = self._mean[index]
        var = self._var[index]
        p = -(x-mean)**2
        numerat = np.exp(p/(2* var))
        denomi = np.sqrt(2*np.pi*var)
        return numerat/denomi

    def predictforone(self, x):
        posteriors = []
        for index, clas in enumerate(self.classes):
            prior = np.log(self.prior[index])
            class_conditional = self._pdf(index, x)  # Gaussian PDF for each feature
            # use log of product = sum of logs
            posterior = np.sum(np.log(class_conditional + 1e-9)) + prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self,X):
        y_pred = [self.predictforone(x) for x in X]
        return np.array(y_pred)

def accuracy(y_pred,y_true):
    return np.sum(y_true == y_pred) / len(y_true)

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data.data
    y = data.target


    X_train,X_test,y_train , y_test = train_test_split(X,y,test_size=0.2)
    model = NaiveBayes()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("accuracy : " , accuracy(predictions,y_test))
