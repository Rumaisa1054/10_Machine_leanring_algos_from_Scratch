from decision_tree import DecisionTreeClassifier
import numpy as np
from collections import Counter
class RandomForestClassifier:
    def __init__(self,n_trees,min_sample_split=2,n_feature=None,max_depth= 10):
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.n_features = n_feature
        self.max_depth = max_depth
        self.trees = []
    def fit(self,X,y):
        #making the trees
        for no in range(self.n_trees):
            tree = DecisionTreeClassifier(self.min_sample_split,self.max_depth,self.n_features)
            x_samples , y_of_these_samples= self.subset_sample(X,y)
            tree.fit(x_samples,y_of_these_samples)
            self.trees.append(tree)
    def subset_sample(self,X,y):
        n_sample = X.shape[0]
        indx_random = np.random.choice(n_sample,n_sample//self.n_trees,replace=True)
        return X[indx_random],y[indx_random]

    def predict(self,X):
        preds = [tr.predict(X) for tr in self.trees]
        # this is going to return array for each prediction from each tree so
        # [[0,0,1], # pred from first tree for 3 samples
        # [0,1,1], # pred from first tree for 3 samples] -> each row has preds for each sample form one tree
        tree_pred = np.swapaxes(preds,0,1) # now each row has preds from each tree for one sample and 0 axis is col - 1 is row
        random_forest_pred_for_each_sample = [self.most_common(pred) for pred in tree_pred]
        return random_forest_pred_for_each_sample
    def most_common(self,y):
        count = Counter(y)
        '''
        most_common(n) returns a list of the n most common elements with their counts.
        count.most_common(1)
        # [('a', 3)]
        '''
        return count.most_common(1)[0][0] # highest most common value just one

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_pred)
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # where y <= 0 replace with -1 else keep 1
    y = np.where(y <= 0, 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(5,)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Accuracy : ", accuracy(predictions, y_test))