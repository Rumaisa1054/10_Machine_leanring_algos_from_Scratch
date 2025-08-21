# SVm for binary classification
import numpy as np
class linear_Reg:
    def __init__(self,learning_rate,iterations):
        # a hyperplane in svm has weight and bais - learning rate
        self.w = None
        self.b = None
        self.alpha = learning_rate
        self.iterations = iterations

    def fit(self,X,y):
        N, features = X.shape

        #set the weights and bais
        self.w = np.random.random(features)
        self.b = np.random.randint(-10,10)

        for i in range(self.iterations):
            pred = np.dot(X, self.w) + self.b
            dw = (2 / N) * np.dot(X.T, (pred-y))
            db = (2 / N) * np.sum(pred - y)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db
        '''
        OR
        
        for _ in range(self.iterations):
            dw = np.zeros(features)
            db = 0
        
            # Loop over each sample
            for idx, x_i in enumerate(X):
                pred_i = np.dot(x_i, self.w) + self.b
                error = pred_i - y[idx]
        
                # Accumulate gradient
                dw += x_i * error
                db += error
        
            # Average and scale
            dw = (2 / N) * dw
            db = (2 / N) * db
        
            # Update weights
            self.w -= self.alpha * dw
            self.b -= self.alpha * db
        '''
    def predict(self,X):
        returner = np.dot(X,self.w) + self.b
        return returner
def accuracy_mse(y_pred,y_true):
    return np.mean((y_true- y_pred) **2)

if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    data = load_diabetes()
    X = data.data
    y = data.target

    # where y <= 0 replace with -1 else keep 1
    y = np.where(y <= 0, -1, 1)

    X_train,X_test,y_train , y_test = train_test_split(X,y,test_size=0.2)
    model = linear_Reg(learning_rate=0.001,iterations=2000)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("Mean squared error : " , accuracy_mse(predictions,y_test))
