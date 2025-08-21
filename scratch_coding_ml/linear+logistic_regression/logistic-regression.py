# SVm for binary classification
import numpy as np

class log_Reg:
    def __init__(self, learning_rate, iterations):
        # a hyperplane in svm has weight and bais - learning rate
        self.w = None
        self.b = None
        self.alpha = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        N, features = X.shape

        # set the weights and bais
        self.w = np.random.random(features)
        # first change
        self.b = 0

        for i in range(self.iterations):
            lin = np.dot(X, self.w) + self.b
            logr = 1  / (1+ np.exp(-lin))
            dw = (1 / N) * np.dot(X.T, (logr - y))
            db = (1 / N) * np.sum(logr - y)

            self.w -= self.alpha * dw
            self.b -= self.alpha * db


    def predict(self, X):
        lin = np.dot(X, self.w) + self.b
        logr = 1 / (1 + np.exp(-lin))
        Ans = [0 if y <=0.5 else 1 for y in logr]
        return Ans


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
    model = log_Reg(learning_rate=0.001, iterations=2000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Accuracy : ", accuracy(predictions, y_test))
