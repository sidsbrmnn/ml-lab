import numpy as np

X = np.array([[2, 9], [1, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)


# Function to return the sigmod of a number
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to return the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


epoch = 1000
learnrate = 0.2
input_layer_size = 2
hidden_layer_size = 3
output_layer_size = 1

# Weights and bias for input and hidden layers
wh = np.random.uniform(size=(input_layer_size, hidden_layer_size))
bh = np.random.uniform(size=(1, hidden_layer_size))

# Weights and bias for hidden and output layers
wout = np.random.uniform(size=(hidden_layer_size, output_layer_size))
bout = np.random.uniform(size=(1, output_layer_size))

# Forward propagation
for i in range(epoch):
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, wout) + bout
    output = sigmoid(output_layer_input)

    # Backpropagation
    EO = y - output
    d_output = EO * sigmoid_derivative(output)

    EH = d_output.dot(wout.T)
    d_hidden = EH * sigmoid_derivative(hidden_layer_output)

    wout += hidden_layer_output.T.dot(d_output) * learnrate
    wh += X.T.dot(d_hidden) * learnrate

# Print prediction
print(output)
