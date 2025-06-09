# Neural Network From Scratch

## Overview

This project implements a simple feedforward neural network from scratch using **Python** and **Numpy**. The neural network is trained on the XOR dataset to learn how to predict the XOR function for binary inputs (0 and 1). This project aims to demonstrate my understanding of fundamental neural network operations, including forward propagation, backpropagation, and training, without using deep learning libraries like **TensorFlow** or **PyTorch**.

### Check out the demo [here](https://drive.google.com/file/d/1YpVzB6VQXRyj1WNqQmXgUmp_3Y6uaZax/view?usp=sharing)!

## Example XOR Inputs and Expected Outputs

Input 1 | Input 2 | Expected Output |
|:-----:|:-------:|:---------------:|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

## Network Architecture

The neural network follows this structure:

• **Input Layer**: 2 neurons (for XOR input values)

• **Hidden Layer**: 3 neurons with a sigmoid activation function

• **Output Layer**: 1 neuron with a sigmoid activation function

## Implementation Details

#### Forward Propagation

1. Compute the **weighted sum** for the hidden layer: 
   $$z_1 = W_1x + b_1$$
   
2. Apply the **sigmoid activation function**: 
   $$\sigma(z_1) = \frac{1}{1 + e^{-z_1}}$$

3. Compute the **weighted sum** for the output layer: 
   $$z_2 = W_2a_1 + b_2$$

4. Apply the **sigmoid activation function** to obtain the final output: 
   $$a_2 = \frac{1}{1 + e^{-z_2}}$$

#### Backpropagation

1. **Calculate the error at the output layer:**
   The error at the output layer is the difference between the actual output (`y`) and the predicted output (`A2`):

   $$\text{error\_output} = y - A2 $$

2. **Calculate the derivative of the sigmoid activation function at the output layer:**
   The derivative of the sigmoid function at the output is used to determine how much the output layer’s weights should be adjusted:
   
   $$dZ2 = \text{error\_output} \times \sigma'(A2)$$
   
   Where $$\( \sigma'(A2) \)$$ is the derivative of the sigmoid function with respect to the output:
   
   $$\sigma'(A2) = A2 \times (1 - A2)$$

3. **Calculate the error at the hidden layer:**
   Using the error from the output layer, the error is propagated backward to the hidden layer. This is done by calculating the derivative of the hidden layer’s activation function and the weights connecting the hidden layer to the output layer:
   
   $$\text{error\_hidden} = dZ2 \cdot W2^T$$

   Where $$\( W2^T \)$$ is the transpose of the weight matrix between the hidden layer and the output layer. 

4. **Calculate the derivative of the sigmoid activation function at the hidden layer:**
   The error at the hidden layer is then multiplied by the derivative of the sigmoid function at the hidden layer:
   
   $$dZ1 = \text{error\_hidden} \times \sigma'(A1)$$

   Where $$\( \sigma'(A1) \)$$ is the derivative of the sigmoid function with respect to the hidden layer’s activations:
   
   $$\sigma'(A1) = A1 \times (1 - A1)$$

5. **Update weights and biases:**
   Once we have the gradients for the weights and biases at both the hidden and output layers, we update the weights and biases using **gradient descent**. 

   - For the output layer:
   
   $$W2 = W2 + \eta \cdot A1^T \cdot dZ2$$
   
   $$b2 = b2 + \eta \cdot \sum{dZ2}$$
   
   Where $$\eta$$ is the learning rate.

   - For the hidden layer:
   
   $$W1 = W1 + \eta \cdot X^T \cdot dZ1$$
   
   $$b1 = b1 + \eta \cdot \sum{dZ1}$$

Where:
- $$\( W1, W2 \)$$ are the weight matrices between the layers.
- $$\( b1, b2 \)$$ are the bias vectors.
- $$\( X \)$$ is the input matrix.
- $$\( A1, A2 \)$$ are the activations of the hidden and output layers, respectively.
- $$\( \eta \)$$ is the learning rate.

In essence, backpropagation allows the network to minimize the error by updating the weights and biases based on the gradients calculated from the loss function.


### Training Process:

- **Epochs**: 10,000 iterations over the training dataset.
- **Learning Rate**: 0.1 for updating weights.
- **Activation Function**: Sigmoid for both hidden and output layers.
- **Optimizer**: Gradient Descent via backpropagation.

## Making predictions

After training, the model allows the user to input custom XOR values (0 or 1) and run them through the trained model. Based on the XOR inputs, the model will predict the output and display what it thinks the result is.

### Example Usage:

```bash
Enter 'exit' at any time to quit.
Enter first binary number (0 or 1): 1
Enter second binary number (0 or 1): 0
Prediction for [1, 0] is: 1

Enter 'exit' at any time to quit.
Enter first binary number (0 or 1): 0
Enter second binary number (0 or 1): 1
Prediction for [0, 1] is: 1
```
#### To Run the Code:

1. Clone the repository.
2. Install the required dependencies with:

   Windows/Linux:
   ```bash
   pip install numpy
   ```
   MacOS:
   ```bash
   pip3 install numpy
   ```
5. Run the Python script to start the XOR prediction model.
6. The model will prompt you to enter your own XOR inputs (either 0 or 1). You can continue to test the model's predictions by providing inputs and receiving output until you type "exit" to quit.






