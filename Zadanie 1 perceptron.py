import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, eta=0.001, n_iter=100):
        self.eta = eta #wspolczynnik uczenia
        self.n_iter = n_iter #Liczba iteracji 
        self.weights = None #wspolczynniki wagowe
        self.errors = [] #zapis bledow

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        
        #trenowanie
        for x in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class MultiClass:
    #perceptron wieloklasowy
    def __init__(self, train_data, train_label):
        self.perceptrons = []
        unique_labels = np.unique(train_label)
        for label in unique_labels:
            binary_labels = np.where(train_label == label, 1, -1)
            perceptron = Perceptron(eta=0.001, n_iter=100)
            perceptron.fit(train_data, binary_labels)
            self.perceptrons.append(perceptron)

    #predykcja 
    def predict(self, X):
        predictions = [np.argmax([perceptron.net_input(xi) for perceptron in self.perceptrons]) for xi in X]
        return np.array(predictions)

def plot_decision_regions(X, y, classifier):
    #wykres
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    markers = ('s', 'x', 'o')
    labels = ['Class 0', 'Class 1', 'Class 2']
    #limity
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #siatka
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap.colors[idx],
                    marker=markers[idx], label=labels[cl])

    plt.legend()
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

def main():
    #zbior danych iris
    iris_data_set = datasets.load_iris()
    X = iris_data_set.data[:, [2, 3]]
    y = iris_data_set.target
    #zbiory treningowe i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    classifier = MultiClass(X_train, y_train)

    predictions = classifier.predict(X_test)
    #wykres
    plot_decision_regions(X=X_test, y=predictions, classifier=classifier)

    plt.xlabel('$x_1$') # $$
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()