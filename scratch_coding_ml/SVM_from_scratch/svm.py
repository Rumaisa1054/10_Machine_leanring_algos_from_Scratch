# SVm for binary classification
import numpy as np
class SVM:
    def __init__(self,lambda_,learning_rate,iterations):
        # a hyperplane in svm has weight and bais - learning rate
        self.w = None
        self.b = None
        self.alpha = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations

    def fit(self,X,y):
        samples, features = X.shape

        '''Support Vector Machine in its basic form is a binary classifier that expects the target labels to be either -1 or +1.'''
        y_new = np.where(y <= 0,-1,1)

        #set the weights and bais
        self.w = np.random.random(features)
        self.b = np.random.randint(-10,10)

        for iter in range(self.iterations):
            for ind, x_i in enumerate(X):
                cond = y_new[ind] * (np.dot(self.w,x_i) - self.b) >= 1
                if cond:
                    # w = w - alpha * 2(lambda *magnitude(w))
                    # b = b - alpha(delta_b) = b
                    self.w = self.w - self.alpha * (2*(self.lambda_)*self.w)
                    self.b = self.b
                else:
                    # w = w - alpha * (2(lambda *magnitude(w)) - (y_i * x_i))
                    # b = b - alpha(delta_b) = b - alpha * y_i
                    self.w = self.w - self.alpha *((2*(self.lambda_) * self.w) - np.dot(x_i,y_new[ind]))
                    self.b = self.b -self.alpha*y_new[ind]


    def predict(self,X):
            returner = np.dot(X,self.w) - self.b
            return np.sign(returner)
def accuracy(y_pred,y_true):
    return np.sum(y_true == y_pred) / len(y_true)
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # where y <= 0 replace with -1 else keep 1
    y = np.where(y <= 0, -1, 1)

    X_train,X_test,y_train , y_test = train_test_split(X,y,test_size=0.2)
    model = SVM(lambda_=0.01,learning_rate=0.001,iterations=1000)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("accuracy : " , accuracy(predictions,y_test))
