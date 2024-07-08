import json
import requests
import spacy

# Define the Word class
class Word:
    def __init__(self, text, lemma, pos, head=None, dep=None):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.head = head
        self.dep = dep
        self.children = []

    def __repr__(self):
        head_text = self.head.text if self.head else None
        return (f"Word(text={self.text}, lemma={self.lemma}, pos={self.pos}, "
                f"head={head_text}, dep={self.dep}, children={[child.text for child in self.children]})")

# Function to load Wikipedia article
def load_wikipedia_article(title):
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    response = requests.get(api_url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract']

# Function to parse text and create an array of sentences with Word instances
def parse_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        words = [Word(token.text, token.lemma_, token.pos_, dep=token.dep_) for token in sent]
        token_to_word = {token: word for token, word in zip(sent, words)}

        # Update head for each word
        for token, word in zip(sent, words):
            if token.head in token_to_word:
                word.head = token_to_word[token.head]

        # Update children for each word
        for word in words:
            word.children = [w for w in words if w.head == word]

        sentences.append(words)
    return sentences

# Load the Wikipedia article for "Artificial Intelligence"
title = "Artificial intelligence"
article_text = load_wikipedia_article(title)

# Parse the text
sentences = parse_text(article_text)

# Show the pairs of governors (parents) and their dependents (children) for the first sentence
print("Governors and their dependents in the first sentence:")
for word in sentences[0]:
    if word.children:
        for child in word.children:
            print(f"Governor: {word.text}, Dependent: {child.text}")
