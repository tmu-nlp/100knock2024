from gensim.models import KeyedVectors
from tqdm import tqdm

def main():
    model_path = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    questions_file = 'questions-words.txt'
    output_file = 'questions-words-add.txt'

    with open(questions_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc='Processing'):
            if line.startswith(':'):
                category = line.split()[1]
            else:
                words = line.split()
                word, cos = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
                f_out.write(f"{category} {' '.join(words)} {word} {cos}\n")

if __name__ == '__main__':
    main()