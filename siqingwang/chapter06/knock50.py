import os
import random

# Define the list of allowed publishers
allowed_publishers = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]

# Function to read and filter articles
def filter_articles(data_dir):
    articles = []
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 8 and parts[3] in allowed_publishers:
                    articles.append((parts[3], parts[1]))  # (publisher, headline)
    return articles

# Function to split data and save to files
def split_and_save_data(data, train_ratio, valid_ratio, train_file, valid_file, test_file):
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)

    train_data = data[:train_size]
    valid_data = data[train_size:train_size+valid_size]
    test_data = data[train_size+valid_size:]

    write_data(train_file, train_data)
    write_data(valid_file, valid_data)
    write_data(test_file, test_data)

def write_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for category, headline in data:
            file.write(f"{category}\t{headline}\n")


if __name__ == '__main__':

    data_dir = "siqingwang/chapter06/newsCorpora.csv"
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test.txt"

    articles = filter_articles(data_dir)

    split_and_save_data(articles, 0.8, 0.1, train_file, valid_file, test_file)

    print("Dataset creation completed.")