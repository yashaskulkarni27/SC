
import numpy as np

# Define the input features and labels
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])  # AND gate

# Initialize weights and bias
weights = np.zeros(inputs.shape[1])
bias = 0
learning_rate = 1

# Fixed increment learning algorithm
def perceptron_training(inputs, labels, weights, bias, learning_rate):
    for _ in range(10):  # Arbitrary number of epochs
        for i, input_vec in enumerate(inputs):
            # Calculate the perceptron output
            linear_output = np.dot(input_vec, weights) + bias
            prediction = 1 if linear_output >= 0 else 0

            # Update weights and bias if there's a prediction error
            error = labels[i] - prediction
            if error != 0:
                weights += learning_rate * error * input_vec
                bias += learning_rate * error
    return weights, bias

# Train the perceptron
final_weights, final_bias = perceptron_training(inputs, labels, weights, bias, learning_rate)

print("Final weights:", final_weights)
print("Final bias:", final_bias)
