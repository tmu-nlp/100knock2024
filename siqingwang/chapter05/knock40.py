import json
import requests
import spacy

# Define the Word class
class Word:
    def __init__(self, text, lemma, pos):
        self.text = text
        self.lemma = lemma
        self.pos = pos

    def __repr__(self):
        return f"Word(text={self.text}, lemma={self.lemma}, pos={self.pos})"

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
        words = [Word(token.text, token.lemma_, token.pos_) for token in sent]
        sentences.append(words)
    return sentences

# Load the Wikipedia article for "Artificial Intelligence"
title = "Artificial intelligence"
article_text = load_wikipedia_article(title)

# Parse the text
sentences = parse_text(article_text)

# Show the object of the first sentence of the body of the article
print("First sentence words:")
for word in sentences[0]:
    print(word)
