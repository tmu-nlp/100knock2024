import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm
from scipy.stats import spearmanr

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calc_cos_sim(row, model):
    try:
        w1v = model[row['Word 1']]
        w2v = model[row['Word 2']]
        return cos_sim(w1v, w2v)
    except KeyError:
        return np.nan

def main():
    model_path = 'GoogleNews-vectors-negative300.bin.gz'
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    csv_file = 'wordsim353/combined.csv'
    combined_df = pd.read_csv(csv_file)

    tqdm.pandas(desc='Calculating cosine similarity')
    combined_df['cos_sim'] = combined_df.progress_apply(calc_cos_sim, axis=1, model=model)

    combined_df = combined_df.dropna(subset=['cos_sim'])

    human_scores = combined_df['Human (mean)'].values
    cos_sim_scores = combined_df['cos_sim'].values

    spearman_corr, _ = spearmanr(human_scores, cos_sim_scores)
    print(f'Spearman correlation: {spearman_corr:.3f}')

if __name__ == '__main__':
    main()

"""
Spearman correlation: 0.700
"""