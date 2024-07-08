# 49

import json


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to extract shortest paths between pairs of nouns
def extract_shortest_paths_between_nouns(data):
    sentences = data['sentences']
    shortest_paths = []

    for sentence in sentences:
        words = sentence['tokens']
        dependencies = sentence['basicDependencies']

        # Collect all nouns and their positions
        nouns = []
        for word in words:
            if word['pos'].startswith('NN'):  # NN for nouns
                nouns.append(word)

        # Iterate through pairs of nouns (i, j) where i < j
        for i in range(len(nouns)):
            for j in range(i + 1, len(nouns)):
                noun_i = nouns[i]
                noun_j = nouns[j]

                # Indices of nouns in the sentence
                index_i = noun_i['index']
                index_j = noun_j['index']

                # Initialize paths
                path_direct = []
                path_common_ancestor = []

                # Function to perform DFS and collect paths
                def dfs_collect_paths(node, path, target_index, paths):
                    path.append(node['word'])

                    if node['index'] == target_index:
                        paths.append(" -> ".join(path))
                        return True

                    if node in children_map:
                        for child in children_map[node]:
                            if dfs_collect_paths(child, path.copy(), target_index, paths):
                                return True
                    return False

                # Create a map of children for efficient traversal
                children_map = {}
                for dep in dependencies:
                    governor_index = dep['governor']
                    dependent_index = dep['dependent']

                    if governor_index in children_map:
                        children_map[governor_index].append(words[dependent_index - 1])  # dependent_index is 1-based
                    else:
                        children_map[governor_index] = [words[dependent_index - 1]]

                # Find direct path from X (noun_i) to Y (noun_j)
                if dfs_collect_paths(noun_i, [], index_j, path_direct):
                    shortest_paths.append(path_direct[0].replace(noun_i['word'], 'X').replace(noun_j['word'], 'Y'))

                # Find common ancestor path between X (noun_i) and Y (noun_j)
                common_ancestor_found = False
                for dep in dependencies:
                    governor_index = dep['governor']
                    dependent_index = dep['dependent']

                    if (governor_index == index_i and dependent_index == index_j) or (
                            governor_index == index_j and dependent_index == index_i):
                        common_ancestor = words[dep['dependent'] - 1]  # dependent_index is 1-based
                        common_ancestor_found = True
                        break

                if common_ancestor_found:
                    path_from_i_to_k = []
                    path_from_k_to_j = []

                    # Path from X (noun_i) to common ancestor (k)
                    dfs_collect_paths(noun_i, [], common_ancestor['index'], path_from_i_to_k)
                    path_from_i_to_k_str = " <- ".join(path_from_i_to_k)

                    # Path from common ancestor (k) to Y (noun_j)
                    dfs_collect_paths(common_ancestor, [], index_j, path_from_k_to_j)
                    path_from_k_to_j_str = " -> ".join(path_from_k_to_j)

                    shortest_paths.append(
                        f"{path_from_i_to_k_str} <- {path_from_k_to_j_str}".replace(noun_i['word'], 'X').replace(
                            noun_j['word'], 'Y'))

    return shortest_paths


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract shortest paths between pairs of nouns
shortest_paths = extract_shortest_paths_between_nouns(data)

# Print extracted shortest paths
print("Extracted Shortest Paths Between Nouns:")
for path in shortest_paths:
    print(path)
