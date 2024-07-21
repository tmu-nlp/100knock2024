import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Parameters
dw = 300  # Dimensionality of word embeddings
dh = 50  # Dimensionality of hidden state
L = 4  # Number of categories (adjust based on your data)
learning_rate = 0.01
epochs = 10


# Example: Function to simulate text to sequences (replace with actual conversion logic)
def text_to_sequences(df, text_column, label_column, mapper):
    sequences = []
    labels = []

    for index, row in df.iterrows():
        words = row[text_column].split()
        sequence = [mapper.get_id(word) for word in words]
        sequences.append(sequence)

        labels.append(row[label_column])

    return sequences, np.array(labels)

class WordToIDMapper:
    def __init__(self):
        self.word_to_id = {}
        self.id_counter = 1  # Start IDs from 1 (0 will be reserved for words occurring less than twice)

    def fit_from_dataframe(self, df, column_name):
        # Extract words from the specified column in the DataFrame
        # 特定されるcolumn(TITLE)から単語をsplit
        data = df[column_name].str.split().sum()  # Split titles into words and flatten into a list

        # Count word frequencies
        # 頻度を計算
        word_counts = Counter(data)

        # Sort words by frequency
        # 頻度から順番を付け
        # based on the second element of each tuple
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Assign IDs to words that occur two or more times
        for word, count in sorted_words:
            if count >= 2:
                self.word_to_id[word] = self.id_counter
                self.id_counter += 1

    def get_id(self, word):
        # Return the ID of a word if it exists, otherwise return 0
        return self.word_to_id.get(word, 0)

# Initialize the mapper
mapper = WordToIDMapper()

# Fit the mapper to your DataFrame
mapper.fit_from_dataframe(train, 'TITLE')

# Convert text data to sequences of word IDs and labels
x_train, y_train = text_to_sequences(train, 'TITLE', 'CATEGORY', mapper)
x_valid, y_valid = text_to_sequences(valid, 'TITLE', 'CATEGORY', mapper)

# Initialize weights and biases
W_hx = np.random.randn(dh, dw)
W_hh = np.random.randn(dh, dh)
b_h = np.zeros((dh, 1))
W_yh = np.random.randn(L, dh)
b_y = np.zeros((L, 1))

# Training loop
for epoch in range(epochs):
    total_loss = 0
    y_preds = []

    # Iterate over training examples
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]

        # Initialize hidden state h_t
        h_t = np.zeros((dh, 1))

        # Forward pass through the RNN
        for t in range(len(x)):
            # Example: Dummy embedding function (replace with actual implementation)
            emb_x = np.random.randn(dw, 1)
            h_t = np.tanh(np.dot(W_hx, emb_x) + np.dot(W_hh, h_t) + b_h)

        # Compute output y using the final hidden state h_T
        y_pred = np.dot(W_yh, h_t) + b_y
        y_probs = np.exp(y_pred) / np.sum(np.exp(y_pred))  # Softmax activation

        # Calculate loss (cross-entropy)
        loss = -np.log(y_probs[y])

        # Backpropagation (not fully implemented)
        # Compute gradients and update parameters

        # Example: Collect predictions for accuracy calculation
        y_preds.append(np.argmax(y_probs))

        # Accumulate total loss
        total_loss += loss

    # Calculate training accuracy
    train_acc = accuracy_score(y_train, y_preds)

    # Print training loss and accuracy for the epoch
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(x_train)}, Training Accuracy: {train_acc}")

    # Optionally, calculate validation loss and accuracy after each epoch
    # val_loss, val_acc = evaluate_model(W_hx, W_hh, W_yh, b_h, b_y, x_valid, y_valid)
    # print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")


# Evaluation function (example)
def evaluate_model(W_hx, W_hh, W_yh, b_h, b_y, x_eval, y_eval):
    total_loss = 0
    y_preds = []

    for i in range(len(x_eval)):
        x = x_eval[i]
        y = y_eval[i]

        h_t = np.zeros((dh, 1))

        # Forward pass through the RNN
        for t in range(len(x)):
            emb_x = np.random.randn(dw, 1)  # Replace with actual embedding function
            h_t = np.tanh(np.dot(W_hx, emb_x) + np.dot(W_hh, h_t) + b_h)

        # Compute output y using the final hidden state h_T
        y_pred = np.dot(W_yh, h_t) + b_y
        y_probs = np.exp(y_pred) / np.sum(np.exp(y_pred))  # Softmax activation

        # Calculate loss (cross-entropy)
        loss = -np.log(y_probs[y])

        # Collect predictions for accuracy calculation
        y_preds.append(np.argmax(y_probs))

        # Accumulate total loss
        total_loss += loss

    # Calculate accuracy
    acc = accuracy_score(y_eval, y_preds)

    return total_loss / len(x_eval), acc
