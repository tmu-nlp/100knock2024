import graphviz
from knock41 import parse_chunks

def visualize_dependency_tree(sentences, sentence_index, output_file, start_index=0, end_index=None):
    """ 指定された文の範囲内の係り受け木を可視化する関数 """
    sentence = sentences[sentence_index]
    dot = graphviz.Digraph(format='png')

    # 日本語フォントの指定
    dot.attr('node', fontname='MS Gothic')

    end_index = end_index if end_index is not None else len(sentence)

    # 指定範囲の文節のみをノードとして追加
    for i, chunk in enumerate(sentence[start_index:end_index]):
        text = ''.join([morph.surface for morph in chunk.morphs if morph.pos != '記号'])
        dot.node(str(i + start_index), text)

    # 指定範囲内の係り先が範囲内にある場合にのみエッジを追加
    for i in range(start_index, end_index):
        chunk = sentence[i]
        if chunk.dst != -1 and start_index <= chunk.dst < end_index:
            dot.edge(str(i), str(chunk.dst))

    dot.render(output_file, cleanup=True)

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    sentence_index = 0  # 例えば、最初の文のインデックス
    output_file = 'knock44'

    # グラフの範囲指定: 例えば、2番目から15番目までの文節を表示
    start_index = 2
    end_index = 16

    if sentences and 0 <= sentence_index < len(sentences):
        visualize_dependency_tree(sentences, sentence_index, output_file, start_index, end_index)
    else:
        print(f"No sentence available at index {sentence_index}. Total sentences: {len(sentences)}")

if __name__ == "__main__":
    main()
