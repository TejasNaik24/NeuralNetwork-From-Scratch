# Neural Network via Numpy

## Overview

This project implements a simple feedforward neural network from scratch using **Python** and **Numpy**. There is no data for training this model and the goal model is to just simply demonstrate my understanding of fundamental neural network operations including forward propagation and conceptual backpropagation, without using deep learning libraries like **TensorFlow** or **PyTorch**.

## Network Architecture

The neural network follows this structure:

• **Input Layer**: 2 neurons (for XOR input values)
• **Hidden Layer**: 2 neurons with a sigmoid activation function
• **Output Layer**: 1 neuron with a sigmoid activation function

## Implementation Details

#### Forward Propagation

1. Compute the **weighted sum** for the hidden layer:
2. Apply the **sigmoid activation function**:
3. Compute the **weighted sum** for the output layer.
4. Apply the **sigmoid activation function** to obtain the final output.

## Backpropagation (Conceptual Explanation) 
• Define a **loss function** (Mean Squared Error or Binary Cross-Entropy).
• Compute **gradients** using the chain rule.
• Explain how **gradient descent** would update **weights**, even though no training is performed

### Running the Code

#### Prerequisites
• Python 3.0x
• Numpy 1.7.0 or higher

## Example XOR Inputs and Expected Outputs

Input 1 | Input 2 | Expected Output |
|:-----:|:-------:|:---------------:|
|0|1|1|
|0|1|1|
|1|0|1|
|1|1|0|

### Contact

For any questions, reach out via GitHub or my email **naik.tejas11@gmail.com**