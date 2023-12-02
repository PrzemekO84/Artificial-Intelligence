import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.special import expit


file_path = "proba4.txt"
data = np.loadtxt(file_path)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    return {
        'W1': np.random.randn(hidden_size, input_size) * 0.01,
        'b1': np.zeros((hidden_size, 1)),
        'W2': np.random.randn(output_size, hidden_size) * 0.01,
        'b2': np.zeros((output_size, 1))
    }


def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters.values()
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = Z2
    return {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}


def compute_cost(A2, Y):
    m = Y.shape[1]
    return (1/m) * np.sum(np.square(A2 - Y))


def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    W1, b1, W2, b2 = parameters.values()
    A1, A2 = cache['A1'], cache['A2']
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * relu_derivative(cache['Z1'])
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def update_parameters(parameters, grads, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate * grads['d' + key]
    return parameters


def predict(X, parameters):
    return forward_propagation(X, parameters)['A2']


def evaluate_model(X, Y, parameters):
    predictions = predict(X, parameters)
    return mean_squared_error(Y, predictions)


def train_neural_network(X, Y, hidden_size, epochs, learning_rate):
    input_size, output_size = X.shape[0], Y.shape[0]
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache['A2'], Y)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        
        validation_loss = evaluate_model(X_val.T.astype(float), y_val.reshape(1, -1).astype(float), parameters)
        test_loss = evaluate_model(X_test.T.astype(float), y_test.reshape(1, -1).astype(float), parameters)

        print(f"Epoch {epoch}, Cost: {cost}, Validation MSE: {validation_loss}, Test MSE: {test_loss}")

    return parameters


hidden_size = 4
epochs = 1000
learning_rate = 0.01


X_train, X_temp, y_train, y_temp = train_test_split(data[:, :-1], data[:, -1], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


trained_parameters = train_neural_network(X_train.T.astype(float), y_train.reshape(1, -1).astype(float), hidden_size, epochs, learning_rate)



X_test = X_test.T.astype(float)
y_test_pred = predict(X_test, trained_parameters)
y_test_pred_tanh = expit(predict(X_test, trained_parameters))  # Apply tanh to the predictions


print("Actual values:")
print(y_test)


X_test = X_test.T.astype(float)
y_test_pred = predict(X_test, trained_parameters)
y_test_pred_tanh = expit(predict(X_test, trained_parameters))  # Apply tanh to the predictions


print("Predicted (ReLU) values:")
print(y_test_pred)
print("Predicted (tanh) values:")
print(y_test_pred_tanh)


plt.figure(figsize=(10, 6))
plt.scatter(X_test.squeeze(), y_test.squeeze(), label='Actual', color='blue', marker='o', s=100)
plt.scatter(X_test.squeeze(), y_test_pred.squeeze(), label='Predicted (ReLU)', color='green', marker='x')
plt.scatter(X_test.squeeze(), y_test_pred_tanh.squeeze(), label='Predicted (tanh)', color='red', marker='o')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
