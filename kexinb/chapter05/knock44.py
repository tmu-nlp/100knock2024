# task44. 係り受け木の可視化
# 与えられた文の係り受け木を有向グラフとして可視化せよ．可視化には，Graphviz等を用いるとよい．

from knock40 import load_file
from knock41 import parse_chunk
from knock42 import chunk_to_text
from graphviz import Digraph


def sentence_to_dot(sentence):
    dot = Digraph(format='png')
    
    for i, chunk in enumerate(sentence):
        chunk_text = chunk_to_text(chunk)
        dot.node(f'chunk_{i}', chunk_text)
    
    for i, chunk in enumerate(sentence):
        if chunk.dst != -1:
            dot.edge(f'chunk_{i}', f'chunk_{chunk.dst}')
    
    return dot

if __name__ == "__main__":
    with open("ai.ja.txt.parsed", "r") as f:
        text = load_file(f)
        chunked_sentences = [parse_chunk(sentence) for sentence in text]
        dot = sentence_to_dot(chunked_sentences[1])
        dot.render("knock44")
        
        # for i, sentence in enumerate(chunked_sentences):
        #     dot = sentence_to_dot(sentence)
        #     dot.render(f'sentence_{i}')