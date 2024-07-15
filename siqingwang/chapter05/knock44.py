import json
import pydot


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to visualize a dependency tree of a sentence
def visualize_dependency_tree(data, sentence_index):
    sentences = data['sentences']

    # Ensure the sentence index is within bounds
    if sentence_index < 0 or sentence_index >= len(sentences):
        print(f"Error: Sentence index {sentence_index} is out of range.")
        return

    sentence = sentences[sentence_index]
    words = sentence['tokens']
    dependencies = sentence['basicDependencies']

    # Create a directed graph
    graph = pydot.Dot(graph_type='digraph')

    # Add nodes for each word
    for word in words:
        node = pydot.Node(str(word['index']), label=f"{word['word']} ({word['lemma']})", shape='ellipse')
        graph.add_node(node)

    # Add edges for dependencies
    for dep in dependencies:
        governor_idx = str(dep['governor'])
        dependent_idx = str(dep['dependent'])
        dep_label = dep['dep']
        edge = pydot.Edge(governor_idx, dependent_idx, label=dep_label)
        graph.add_edge(edge)

    # Save the DOT source code to a file (optional)
    dot_file_path = f"sentence_{sentence_index + 1}_dependency_tree.dot"
    graph.write_dot(dot_file_path)
    print(f"DOT source saved to: {dot_file_path}")

    # Render the graph (optional)
    image_file_path = f"sentence_{sentence_index + 1}_dependency_tree.png"
    graph.write_png(image_file_path)
    print(f"Graph image saved to: {image_file_path}")

    # Display the graph (optional)
    # You can display the graph inline in Jupyter Notebook or other environments
    # graph.write_svg(f"sentence_{sentence_index + 1}_dependency_tree.svg")

    print(f"Visualized dependency tree for sentence {sentence_index + 1}")


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Visualize the dependency tree for a specific sentence (e.g., sentence index 0)
sentence_index = 0  # Change this index to visualize a different sentence
visualize_dependency_tree(data, sentence_index)
