import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.w_ = np.random.normal(0, 0.01, 1 + X.shape[1])
        self.cost_ = []

        for x in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250))).astype(float)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    

class MultiClassLogisticRegression(object):

    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifiers = []


    def fit(self, X, y):
        unique_classes = np.unique(y)

        for class_label in unique_classes:
            binary_y = np.where(y == class_label, 1, 0)
            classifier = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            classifier.fit(X, binary_y)
            self.classifiers.append((class_label, classifier))

        return self

    def predict(self, X):
        predictions = [classifier.net_input(X) for _, classifier in self.classifiers]
        predictions = np.array(predictions).T
        return np.argmax(predictions, axis=1)

def main():
    iris_data_set = datasets.load_iris()
    X = iris_data_set.data[:, [2, 3]]
    y = iris_data_set.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    
    sc = StandardScaler()
    X_train_std, X_test_std = sc.fit_transform(X_train), sc.transform(X_test)

    mclr = MultiClassLogisticRegression(eta=0.05, n_iter=1000, random_state=1)
    mclr.fit(X_train_std, y_train)

    predictions = mclr.predict(X_test_std)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("Accuracy:", accuracy)

    plot_decision_regions(X=X_test_std, y=y_test, clf=mclr)
    plt.xlabel('Petal length (standardized)')
    plt.ylabel('Petal width (standardized)')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()