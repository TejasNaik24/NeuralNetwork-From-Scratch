import numpy as np

# Defining the network's architecture
input_size = 2
hidden_size = 2
output_size = 1

#np.random.seed(42)  # For reproducibility
# Initialize weights and biases and giv
W1 = np.random.randn(hidden_size, input_size)
b1 = np.random.randn(hidden_size, 1)
W2 = np.random.randn(output_size, hidden_size)
b2 = np.random.randn(output_size, 1)

# Activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Loss function (Mean Squared Error)
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# XOR inputs (hardcoded, not for training)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Shape (2,4)
Y = np.array([[0], [1], [1], [0]])  # Shape (4,1)

# --- Forward Pass ---
Z1, A1, Z2, A2 = forward_propagation(X)

# Print output before learning
print("Initial Neural Network Output:\n", A2)

# --- Initial Loss ---
initial_loss = mean_squared_error(A2, Y.T)
print("Initial Loss:", initial_loss)

'''

__________understanding Backpropagation code_____________

 Backpropagation (just one pass)
def backpropagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[1]  # Number of examples

     Output layer gradients
    dZ2 = A2 - Y.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

     Hidden layer gradients
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

 --- Backpropagation ---
dW1, db1, dW2, db2 = backpropagation(X, Y, Z1, A1, Z2, A2)

 --- One-time update (optional for demo) ---
learning_rate = 0.1
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2

''' 