from tqdm import tqdm

def main():
    input_file = 'questions-words-add.txt'

    with open(input_file, 'r', encoding='utf-8') as f:
        sem_cnt = 0
        sem_cor = 0
        syn_cnt = 0
        syn_cor = 0

        for line in tqdm(f, desc='Processing'):
            line = line.split()
            if not line[0].startswith('gram'):
                sem_cnt += 1
                if line[4] == line[5]:
                    sem_cor += 1
            else:
                syn_cnt += 1
                if line[4] == line[5]:
                    syn_cor += 1

    print(f'意味的アナロジー正解率: {sem_cor/sem_cnt:.3f}')
    print(f'文法的アナロジー正解率: {syn_cor/syn_cnt:.3f}')

if __name__ == '__main__':
    main()


"""
意味的アナロジー正解率: 0.731
文法的アナロジー正解率: 0.740
"""
