import numpy as np

# Parameters
dw = 300  # Dimensionality of word embeddings
dh = 50  # Dimensionality of hidden state
L = 4  # Number of categories

# Initialize random weights and biases
W_hx = np.random.randn(dh, dw)
W_hh = np.random.randn(dh, dh)
b_h = np.zeros((dh, 1))

W_yh = np.random.randn(L, dh)
b_y = np.zeros((L, 1))

# Example sequence of word IDs (replace with actual data)
x = np.array([1, 3, 2, 4, 1])

# Initialize hidden state h_t
h_t = np.zeros((dh, 1))


# Define activation function (e.g., tanh)
def tanh(x):
    return np.tanh(x)


# RNN cell function
def rnn_cell(x_t, h_prev):
    # Embedding: Convert one-hot vector to dense vector (not implemented explicitly here)
    emb_x = np.random.randn(dw, 1)  # Replace with actual embedding function

    # RNN cell computation
    h_next = tanh(np.dot(W_hx, emb_x) + np.dot(W_hh, h_prev) + b_h)

    return h_next


# Forward pass through the RNN
for t in range(len(x)):
    h_t = rnn_cell(x[t], h_t)

# Compute y using the final hidden state h_T
y = np.dot(W_yh, h_t) + b_y
y_prob = np.exp(y) / np.sum(np.exp(y))  # Softmax activation

print("Output probabilities (y):")
print(y_prob)
