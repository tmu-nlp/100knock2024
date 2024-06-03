from knock41 import parse_chunks

def extract_noun_to_root_paths(sentences):
    """
    文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出する関数
    """
    paths = []
    for sentence in sentences:
        for chunk in sentence:
            if any(morph.pos == '名詞' for morph in chunk.morphs):
                path = []
                current_chunk = chunk
                while current_chunk.dst != -1:
                    path.append(str(current_chunk))
                    current_chunk = sentence[current_chunk.dst]
                path.append(str(current_chunk))
                paths.append(' -> '.join(path))
    return paths

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    paths = extract_noun_to_root_paths(sentences)
    output_file = 'knock48.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(path + '\n')
    print(f"Output has been saved to {output_file}")

if __name__ == "__main__":
    main()