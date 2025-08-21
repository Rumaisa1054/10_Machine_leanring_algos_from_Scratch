import numpy as np
from collections import Counter
class Node:
    def __init__(self,f=None,t=None,l=None,r=None,*,val = None): # writing the * before an argument - if I want to pass a value i would have to explkiccitly mention the val = xyz
        self.feature = f
        self.threshold = t
        self.left = l
        self.right = r
        self.value = val
    def is_leaf(self):
        return self.value is not None
class DecisionTreeClassifier:
    def __init__(self,min_sample_split = 2,max_depth=100,n_features = None):
        # stopping criteria
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        # root of decision tree
        self.root = None
    # recursive function to create the tree
    def entropy_cal(self,e_):
        # E(x) = - sum( p(X) * log_base_2(p(X))
        # p(X) = no of times this class label has occured in this node / no of nodes = #x/n
        # hist = np.bincount(y)
        histogram = [0] * (max(e_)+1) # indeses are values - the elemnents are frequency
        for item in e_:
            histogram[item] += 1
        hist = np.array(histogram)
        p_x = hist / len(e_)
        summ = 0
        for p in p_x:
            summ += p * np.log2(p) if p > 0 else 0
        return -summ
    def split(self,col,thres):
        left_indxs = np.argwhere(col <= thres).flatten()
        right_indxs = np.argwhere(col > thres).flatten()
        return left_indxs,right_indxs
    def information_gain_calc(self,y,X_column,thres):
        # IG = E(parent) - [weighted_avg] E(child) where E = entropy of
        # Step : 01: parent entropy calculation
        parent_ent = self.entropy_cal(y)
        # step : 02 create children
        left_inds , right_inds = self.split(X_column,thres)
        if len(left_inds) == 0 or len(right_inds) == 0: return 0

        # step : 03 caculated weighted avg * entropy of child
        n = len(y)
        n_l = len(left_inds)
        n_r = len(right_inds)
        e_l = self.entropy_cal(y[left_inds])
        e_r = self.entropy_cal(y[right_inds])
        child_ent = n_l/n * e_l + n_r/n * e_r
        information_gain = parent_ent - child_ent
        return information_gain

    def best_split_finder(self,X,y,features_indexes):
        # we are going to find all possible threshold that are out there
        # all possible split_features that oculd be used and select the best one
        best_gain = 0
        split_feature_index = split_threshold = None

        for ind in features_indexes:
            X_column = X[:,ind]
            for thres in (np.unique(X_column)):
                gain = self.information_gain_calc(y,X_column,thres)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = thres
                    split_feature_index = ind

        return split_feature_index,split_threshold


    def grow_tree(self,X,y,depth):
        # stopping criteria used in recursion
        n_samples,no_features_now = X.shape
        n_classes = len(np.unique(y))
        # checkc the stopping critera
        if depth >= self.max_depth or n_classes == 1 or n_samples <= self.min_samples_split:
            leaf_value = self.most_common(y)
            return Node(val=leaf_value)
        # the no of features I have right now in x and then select the self.n_featuresf from them which are to be checked
        features_indexes = np.random.choice(no_features_now,self.n_features,replace=False) # false that donot replace it -- it would lead to duplicates indexes of features
        # find the best split
        # features_indexes = indexes of features_I_want_it_to_consider_for_split
        best_feature_ind,thresh_best = self.best_split_finder(X,y,features_indexes)
        # create the child nodes
        left_inds,right_inds = self.split(X[:,best_feature_ind],thresh_best)
        left = self.grow_tree(X[left_inds,:] , y[left_inds] , depth + 1)
        right = self.grow_tree(X[right_inds,:],y[right_inds], depth + 1)
        n = Node(f = best_feature_ind,t=thresh_best,l = left,r = right)
        return n
    def most_common(self,y):
        count = Counter(y)
        '''
        most_common(n) returns a list of the n most common elements with their counts.
        count.most_common(1)
        # [('a', 3)]
        '''
        return count.most_common(1)[0][0] # highest most common value just one

    def fit(self,X,y):
        # n_features : no of features in actual
        # check if no of features to be considered in each time splitting is not more than total features
        self.n_features = X.shape[1]  if not self.n_features else min(self.n_features,X.shape[1])
        # creates the tree and return the root
        self.root = self.grow_tree(X,y,0)

    def traverser(self,y_i,node):
        if node.is_leaf() == True:
            return node.value
        split_f = node.feature
        if y_i[split_f] <= node.threshold:
            return self.traverser(y_i, node.left)
        else:
            return self.traverser(y_i, node.right)


    def predict(self,y):
        return np.array([self.traverser(y_i,self.root) for y_i in y])

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
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Accuracy : ", accuracy(predictions, y_test))