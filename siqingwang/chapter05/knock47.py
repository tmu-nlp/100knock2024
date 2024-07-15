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


# Function to extract facts from sentences in passive voice
def extract_passive_facts(data):
    sentences = data['sentences']
    passive_facts = []

    for sentence in sentences:
        words = sentence['tokens']
        dependencies = sentence['enhancedPlusPlusDependencies']  # Using enhanced++ dependencies

        # Check for passive voice structure
        has_passive_structure = False
        for dep in dependencies:
            if dep['dep'] == 'ROOT' and dep['governor'] != 0:
                verb_index = dep['governor']
                verb_token = next((word for word in words if word['index'] == verb_index), None)

                if verb_token and verb_token['pos'] == 'VBN':  # Check if verb is past participle (passive voice)
                    has_passive_structure = True
                    break

        if has_passive_structure:
            subject = None
            predicate = None
            obj = None

            # Find subject (nsubjpass or csubjpass)
            for subj_dep in dependencies:
                if subj_dep['dep'] in ['nsubjpass', 'csubjpass']:
                    subj_token = next((word for word in words if word['index'] == subj_dep['dependent']), None)
                    if subj_token:
                        visited = set()
                        subject = extract_phrase(words, dependencies, subj_token, visited)
                        break

            # Find predicate (ROOT)
            for root_dep in dependencies:
                if root_dep['dep'] == 'ROOT':
                    verb_index = root_dep['governor']
                    verb_token = next((word for word in words if word['index'] == verb_index), None)
                    if verb_token:
                        verb_word = verb_token['word']
                        predicate = f"{verb_word}-{root_dep['dep']}"
                        break

            # Find object (dobj, prep_x)
            for obj_dep in dependencies:
                if obj_dep['dep'] in ['dobj', 'prep_in', 'prep_as']:
                    obj_token = next((word for word in words if word['index'] == obj_dep['dependent']), None)
                    if obj_token:
                        visited = set()
                        obj = extract_phrase(words, dependencies, obj_token, visited)
                        break

            # Append fact tuple if all components are found
            if subject and predicate and obj:
                passive_facts.append((subject, predicate, obj))

    return passive_facts


# Path to your ai.en.txt.json file
json_file_path = '/content/drive/My Drive/NLP/ai/ai.en.txt.json'

# Load JSON data
data = load_json_file(json_file_path)

# Extract passive voice facts
passive_facts = extract_passive_facts(data)

# Print extracted facts
print("Extracted Passive Voice Facts:")
for fact in passive_facts:
    print(fact)
