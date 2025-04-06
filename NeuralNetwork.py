import numpy as np

# Defining the network's architecture
input_size = 2
hidden_size = 2
output_size = 1

np.random.seed(42)  # For reproducibility
W1 = np.random.randn(hidden_size, input_size)  # Weights for input -> hidden
b1 = np.random.randn(hidden_size, 1)           # Bias for hidden layer
W2 = np.random.randn(output_size, hidden_size) # Weights for hidden -> output
b2 = np.random.randn(output_size, 1)           # Bias for output layer

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation function
def forward_propagation(X):
    # Hidden layer computations
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)  # Apply activation function
    
    # Output layer computations
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # Final output
    
    return A1, A2  # Return both A1 (hidden layer output) and A2 (final output)

# XOR inputs
XOR_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Shape (2,4)

# Expected XOR outputs
y = np.array([[0], [1], [1], [0]])

# Compute output
A1, output = forward_propagation(XOR_inputs)

print("Neural Network Output:\n", output)