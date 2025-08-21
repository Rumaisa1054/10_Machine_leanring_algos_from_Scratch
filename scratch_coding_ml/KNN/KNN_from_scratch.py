import numpy as np
from collections import defaultdict
class KNN:
    def __init__(self,n_neighbors,prob):
        self.K = n_neighbors
        self.X_train = None
        self.y_train = None
        self.prob = prob

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict_for_one(self,X):
        def eucledian_distance(x1, x2):
            return np.sqrt(np.sum((x1 - x2) ** 2))
        distances = []
        for i,x_i in enumerate(self.X_train):
            # storing the index of the datapoint and the distance
            st = (eucledian_distance(x_i,X),i)
            distances.append(st)

        # sorting in ascending order
        distances.sort()

        #getting k closest neighbors
        k_neigh = distances[:self.K]
        indexes = [i for dist,i in k_neigh]

        # getting the majority vote
        if self.prob == "Classification":
            dicte = defaultdict(int)
            for item in indexes:
                label = self.y_train[item]
                dicte[label] +=1
            max_key = max(dicte, key=dicte.get)
            return max_key
        else:
            # self.prob == "Regression
            total = 0
            for item in indexes:
                total += self.y_train[item]
            return total/self.K

    def predict(self,X):
        predictions = [self.predict_for_one(x) for x in X]
        return predictions

def accuracy(y_pred,y_true):
    return np.sum(y_true == y_pred) / len(y_true)

if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data.data
    y = data.target


    X_train,X_test,y_train , y_test = train_test_split(X,y,test_size=0.2)
    model = KNN(prob="Classification",n_neighbors=4)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("accuracy : " , accuracy(predictions,y_test))
