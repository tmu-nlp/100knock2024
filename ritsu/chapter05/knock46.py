from knock41 import parse_chunks

def extract_verb_frame_information(sentences):
    """ 動詞の格フレーム情報を抽出する関数 """
    frames = []
    for sentence in sentences:
        for chunk in sentence:
            verbs = [morph.base for morph in chunk.morphs if morph.pos == '動詞']
            if verbs:  # 文節に動詞が含まれる場合
                first_verb = verbs[0]  # 最左の動詞の基本形を取得
                particles = []
                phrases = []
                for src in chunk.srcs:
                    src_chunk = sentence[src]
                    particle = [morph.base for morph in src_chunk.morphs if morph.pos == '助詞']
                    phrase = ''.join([morph.surface for morph in src_chunk.morphs if morph.pos != '記号'])
                    if particle:
                        particles.append(particle[-1])  # 各係り元文節の最後の助詞を取得
                        phrases.append(phrase)  # その文節全体を取得
                if particles:
                    # 辞書順にソート
                    paired = sorted(zip(particles, phrases))
                    sorted_particles, sorted_phrases = zip(*paired)
                    frame = f"{first_verb}\t{' '.join(sorted_particles)}\t{' '.join(sorted_phrases)}"
                    frames.append(frame)
    return frames

def main():
    file_path = 'ai.ja.txt.parsed'
    sentences = parse_chunks(file_path)
    verb_frames = extract_verb_frame_information(sentences)
    output_file = 'knock46.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for frame in verb_frames:
            f.write(frame + '\n')
    print(f"Output has been saved to {output_file}")

if __name__ == "__main__":
    main()
