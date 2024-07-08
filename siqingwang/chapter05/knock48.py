# 48
import json


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to extract paths from the root to nouns
def extract_paths_to_nouns(data):
    sentences = data['sentences']
    paths_to_nouns = []

    for sentence in sentences:
        words = sentence['tokens']
        dependencies = sentence['basicDependencies']

        # Find the root (typically the verb)
        root = None
        for dep in dependencies:
            if dep['dep'] == 'ROOT' and dep['governor'] != 0:
                root_token = next((word for word in words if word['index'] == dep['governor']), None)
                if root_token:
                    root = root_token['word']
                    break

        if not root:
            continue

        # Collect all nouns in the sentence
        nouns = []
        for word in words:
            if word['pos'].startswith('NN'):  # NN for nouns
                nouns.append(word)

        # Function to perform DFS and collect paths to nouns
        def dfs_collect_paths(node, path, paths):
            path.append(node['word'])

            if node in children_map:
                for child in children_map[node]:
                    dfs_collect_paths(child, path.copy(), paths)
            else:
                paths.append(" -> ".join(path))

        # Create a map of children for efficient traversal
        children_map = {}
        for dep in dependencies:
            governor_index = dep['governor']
            dependent_index = dep['dependent']

            if governor_index in children_map:
                children_map[governor_index].append(words[dependent_index - 1])  # dependent_index is 1-based
            else:
                children_map[governor_index] = [words[dependent_index - 1]]

        # Extract paths from root to each noun
        for noun in nouns:
            path_to_noun = []
            noun_index = noun['index']
            if noun_index in children_map:
                dfs_collect_paths(noun, [], path_to_noun)
                paths_to_nouns.extend(path_to_noun)

    return paths_to_nouns


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract paths from the root to nouns
paths_to_nouns = extract_paths_to_nouns(data)

# Print extracted paths
print("Extracted paths from root to nouns:")
for path in paths_to_nouns:
    print(path)
