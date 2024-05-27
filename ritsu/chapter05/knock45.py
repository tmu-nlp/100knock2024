from knock41 import parse_chunks

def extract_verb_patterns(sentences):
    """ 動詞の格パターンを抽出する関数 """
    patterns = []
    for sentence in sentences:
        for chunk in sentence:
            verbs = [morph.base for morph in chunk.morphs if morph.pos == '動詞']
            if verbs:  # 文節に動詞が含まれる場合
                first_verb = verbs[0]  # 最左の動詞の基本形を取得
                particles = []
                for src in chunk.srcs:
                    src_chunk = sentence[src]
                    particle = [morph.base for morph in src_chunk.morphs if morph.pos == '助詞']
                    if particle:
                        particles.append(particle[-1])  # 各係り元文節の最後の助詞を取得
                if particles:
                    particles_sorted = sorted(particles)  # 助詞を辞書順に並べ替え
                    pattern = f"{first_verb}\t{' '.join(particles_sorted)}"
                    patterns.append(pattern)
    return patterns

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    verb_patterns = extract_verb_patterns(sentences)
    output_file = 'knock45.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pattern in verb_patterns:
            f.write(pattern + '\n')
    print(f"Output has been saved to {output_file}")

if __name__ == "__main__":
    main()
