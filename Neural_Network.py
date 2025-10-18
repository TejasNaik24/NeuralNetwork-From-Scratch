import numpy as np

# Sigmoid Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid Activation function derivative
def sigmoid_derivative(a):
    return a * (1 - a)

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Defining the architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #creating random weights and biases
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forwardProp(self, X):
        #forward passing inputs through model
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    # Binary Cross-Entropy loss function
    def binary_cross_entropy(self, y_true, y_pred, epsilon=1e-15):
        #preventing log(0) error
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def backwardProp(self, X, y):
        # Error at the output
        error_output = y - self.A2
        dZ2 = error_output * sigmoid_derivative(self.A2)

        # Error at the hidden layer
        error_hidden = dZ2.dot(self.W2.T)
        dZ1 = error_hidden * sigmoid_derivative(self.A1)

        # Updating the weights and biases
        self.W2 += self.A1.T.dot(dZ2) * self.learning_rate
        self.b2 += np.sum(dZ2, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(dZ1) * self.learning_rate
        self.b1 += np.sum(dZ1, axis=0, keepdims=True) * self.learning_rate


    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            self.forwardProp(X)

            # Compute the loss
            loss = self.binary_cross_entropy(y, self.A2)

              # Backpropagation
            self.backwardProp(X, y)

            yield epoch, loss


    #making the predictions
    def predict(self, X):
       output = self.forwardProp(X)
       return (output > 0.5).astype(int)