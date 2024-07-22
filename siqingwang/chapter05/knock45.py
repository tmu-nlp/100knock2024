import json


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to extract tuples of (subject, verb, object)
def extract_svo_tuples(data):
    sentences = data['sentences']
    svo_tuples = []

    for sentence in sentences:
        words = sentence['tokens']
        dependencies = sentence['basicDependencies']

        # Initialize variables to store subject, verb, and object
        subject = None
        verb = None
        obj = None

        # Find the root verb (predicate in past tense)
        root_verb = None
        for dep in dependencies:
            if dep['dep'] == 'ROOT' and words[dep['dependent'] - 1]['pos'] == 'VBD':  # VBD for past tense verb
                root_verb = words[dep['dependent'] - 1]
                break

        if root_verb is None:
            continue

        # Find the nominal subject (nsubj) of the root verb
        for dep in dependencies:
            if dep['governor'] == root_verb['index'] and dep['dep'] == 'nsubj':
                subject = words[dep['dependent'] - 1]['word']
                break

        if subject is None:
            continue

        # Find the direct object (dobj) of the root verb
        for dep in dependencies:
            if dep['governor'] == root_verb['index'] and dep['dep'] == 'dobj':
                obj = words[dep['dependent'] - 1]['word']
                break

        if obj is None:
            continue

        # Append the tuple (subject, verb, object) to results
        svo_tuples.append((subject, root_verb['word'], obj))

    return svo_tuples


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract tuples (subject, verb, object)
svo_tuples = extract_svo_tuples(data)

# Print extracted tuples
print("Extracted SVO Tuples:")
for tuple in svo_tuples:
    print(tuple)
