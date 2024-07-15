import json


# Function to load JSON data from file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Function to recursively extract phrases from dependency tree
def extract_phrase(tokens, dependencies, start_token, visited):
    phrase_tokens = [start_token]
    visited.add(start_token['index'])

    for dep in dependencies:
        if dep['governor'] == start_token['index'] and dep['dep'] in ['compound', 'amod', 'det', 'nummod']:
            dependent_token = next((token for token in tokens if token['index'] == dep['dependent']), None)
            if dependent_token and dependent_token['index'] not in visited:
                phrase_tokens.append(dependent_token)
                visited.add(dependent_token['index'])
                extract_phrase(tokens, dependencies, dependent_token, visited)

    # Sort tokens by index before composing the phrase
    phrase_tokens.sort(key=lambda x: x['index'])
    phrase_text = ' '.join(token['word'] for token in phrase_tokens)
    return phrase_text


# Function to extract subject-verb-object triples from dependency trees
def extract_svo_tuples(data):
    sentences = data['sentences']
    svo_tuples = []

    for sentence in sentences:
        words = sentence['tokens']
        dependencies = sentence['enhancedPlusPlusDependencies']  # Using enhanced++ dependencies

        for dep in dependencies:
            if dep['dep'] == 'ROOT' and dep['governor'] != 0:
                verb_index = dep['governor']
                verb_token = next((word for word in words if word['index'] == verb_index), None)

                if verb_token and verb_token['pos'].startswith('V'):  # Check if verb and in past tense
                    verb_lemma = verb_token['lemma']
                    verb_word = verb_token['word']
                    past_tense = ('past' in verb_token['features'].split('|'))

                    if past_tense:
                        subject = None
                        obj = None

                        # Find subject (nominal subjects and noun clauses)
                        for subj_dep in dependencies:
                            if subj_dep['governor'] == verb_index and (
                                    subj_dep['dep'] in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']):
                                subj_token = next((word for word in words if word['index'] == subj_dep['dependent']),
                                                  None)
                                if subj_token:
                                    visited = set()
                                    subject = extract_phrase(words, dependencies, subj_token, visited)
                                    break

                        # Find object (direct objects)
                        for obj_dep in dependencies:
                            if obj_dep['governor'] == verb_index and obj_dep['dep'] == 'dobj':
                                obj_token = next((word for word in words if word['index'] == obj_dep['dependent']),
                                                 None)
                                if obj_token:
                                    visited = set()
                                    obj = extract_phrase(words, dependencies, obj_token, visited)
                                    break

                        # Append SVO tuple if subject and object are found
                        if subject and obj:
                            svo_tuples.append((subject, verb_word, obj))

    return svo_tuples


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract subject-verb-object triples
svo_tuples = extract_svo_tuples(data)

# Print extracted tuples
print("Extracted Subject-Verb-Object (SVO) Tuples:")
for svo in svo_tuples:
    print(svo)
