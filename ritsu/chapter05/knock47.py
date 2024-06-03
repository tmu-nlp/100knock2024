from knock41 import parse_chunks

def extract_sa_hen_verb_constructions(sentences):
    """
    サ変接続名詞が「を」を伴い、動詞に係る構文を抽出する関数
    """
    results = []
    for sentence in sentences:
        for chunk in sentence:
            verbs = [morph.base for morph in chunk.morphs if morph.pos == '動詞']
            if verbs:
                for src in chunk.srcs:
                    src_chunk = sentence[src]
                    if any(morph.base == 'を' and morph.pos == '助詞' for morph in src_chunk.morphs):
                        sa_hen_nouns = [morph.base for morph in src_chunk.morphs if morph.pos == '名詞' and morph.pos1 == 'サ変接続']
                        if sa_hen_nouns:
                            verb = verbs[0]  # 最左の動詞
                            predicate = f"{sa_hen_nouns[0]}を{verb}"  # 構文を形成
                            
                            particles = []
                            terms = []
                            for src2 in chunk.srcs:
                                src2_chunk = sentence[src2]
                                if src2_chunk != src_chunk:
                                    particles.extend([morph.base for morph in src2_chunk.morphs if morph.pos == '助詞'])
                                    terms.append(''.join(morph.surface for morph in src2_chunk.morphs if morph.pos != '記号'))
                            
                            if particles and terms:
                                particles_sorted, terms_sorted = zip(*sorted(zip(particles, terms)))
                                result = f"{predicate}\t{' '.join(particles_sorted)}\t{' '.join(terms_sorted)}"
                                results.append(result)
    return results

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    constructions = extract_sa_hen_verb_constructions(sentences)
    output_file = 'knock47.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for construction in constructions:
            f.write(construction + '\n')
    print(f"Output has been saved to {output_file}")

if __name__ == "__main__":
    main()