import json


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to extract verb governors and noun dependents
def extract_verb_governors_and_noun_dependents(data):
    sentences = data['sentences']
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}:")
        dependencies = sentence['basicDependencies']
        tokens = sentence['tokens']

        # Mapping tokens by their index
        token_map = {token['index']: token for token in tokens}

        try:
            for dep in dependencies:
                governor_index = dep['governor']
                dependent_index = dep['dependent']

                governor_token = token_map.get(governor_index)
                dependent_token = token_map.get(dependent_index)

                if governor_token and governor_token['pos'].startswith('V'):
                    verb = governor_token
                    governors = []
                    noun_dependents = []

                    if dep['dep'] == 'ROOT':
                        for d in dependencies:
                            if d['governor'] == governor_index:
                                child_token = token_map.get(d['dependent'])
                                if child_token and child_token['pos'].startswith('N'):
                                    noun_dependents.append(child_token)

                    # Print verb governors and their noun dependents
                    if noun_dependents:
                        print(f"  Verb: {verb['word']} ({verb['lemma']})")
                        print(f"    Governors: {[gov['word'] for gov in governors]}")
                        print(f"    Noun dependents: {[nd['word'] for nd in noun_dependents]}")
                        print()
        except KeyError as e:
            print(f"KeyError: {e} in sentence {i + 1}")


# Path to your existing ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract and display verb governors and noun dependents
extract_verb_governors_and_noun_dependents(data)
