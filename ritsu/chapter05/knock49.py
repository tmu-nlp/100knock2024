from knock41 import parse_chunks

def extract_noun_paths(sentence):
    """
    文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出する関数
    """
    noun_phrases = []
    for i, chunk in enumerate(sentence):
        if '名詞' in [morph.pos for morph in chunk.morphs]:
            noun_phrases.append((i, chunk))

    for i, (noun_i, chunk_i) in enumerate(noun_phrases):
        for noun_j, chunk_j in noun_phrases[i + 1:]:
            path_i = []
            path_j = []
            current_chunk_i = chunk_i
            current_chunk_j = chunk_j
            while current_chunk_i.dst != -1 or current_chunk_j.dst != -1:
                if current_chunk_i.dst == -1:
                    path_j.append(str(current_chunk_j))
                    current_chunk_j = sentence[current_chunk_j.dst]
                elif current_chunk_j.dst == -1:
                    path_i.append(str(current_chunk_i))
                    current_chunk_i = sentence[current_chunk_i.dst]
                else:
                    path_i.append(str(current_chunk_i))
                    path_j.append(str(current_chunk_j))
                    current_chunk_i = sentence[current_chunk_i.dst]
                    current_chunk_j = sentence[current_chunk_j.dst]
                    if current_chunk_i == current_chunk_j:
                        break

            if current_chunk_i == current_chunk_j:
                path_i.append(str(current_chunk_i))
                path_i = ['X' if i == noun_i else chunk for i, chunk in enumerate(path_i)]
                path_j = ['Y' if i == noun_j else chunk for i, chunk in enumerate(path_j[::-1])]
                path = ' | '.join([' -> '.join(path_i), ' -> '.join(path_j)])
            else:
                path_i = ['X' if i == noun_i else chunk for i, chunk in enumerate(path_i)]
                path_j = ['Y' if i == noun_j else chunk for i, chunk in enumerate(path_j)]
                path = ' | '.join([' -> '.join(path_i), ' -> '.join(path_j)])

            yield path

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    output_file = 'knock49.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for path in extract_noun_paths(sentence):
                f.write(path + '\n')
    print(f"Output has been saved to {output_file}")

if __name__ == "__main__":
    main()