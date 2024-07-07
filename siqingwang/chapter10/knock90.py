import nltk
from nltk.tokenize import word_tokenize
import os
from sudachipy import tokenizer, dictionary

# NLTKデータのダウンロード
nltk.download('punkt')

# SudachiPyの初期化
tokenizer_obj = dictionary.Dictionary().create()

# 日本語の形態素トークン化
# SudachiPyを使用して日本語テキストを形態素解析
def tokenize_with_sudachipy(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tokenized_lines = []
    for line in lines:
        tokens = tokenizer_obj.tokenize(line)
        tokens = [morph.surface() for morph in tokens]
        tokenized_lines.append(' '.join(tokens))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tokenized_lines))

# 英語の単語トークン化
# NLTKの word_tokenize を使用して英語テキストをトークン化
def tokenize_english(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    tokens = word_tokenize(data)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens))

# トークン化のためのファイルリスト（from KFTTデータセット）
japanese_files_to_tokenize = [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'kyoto-train.tok.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'kyoto-dev.tok.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'kyoto-test.tok.ja')
]

english_files_to_tokenize = [
    ('kftt-data-1.0/data/orig/kyoto-train.en', 'kyoto-train.tok.en'),
    ('kftt-data-1.0/data/orig/kyoto-dev.en', 'kyoto-dev.tok.en'),
    ('kftt-data-1.0/data/orig/kyoto-test.en', 'kyoto-test.tok.en')
]

# 日本語のファイルをトークン化
for input_file, output_file in japanese_files_to_tokenize:
    if os.path.exists(input_file):
        tokenize_with_sudachipy(input_file, output_file)
    else:
        print(f"File not found: {input_file}")

# 英語のファイルをトークン化
for input_file, output_file in english_files_to_tokenize:
    if os.path.exists(input_file):
        tokenize_english(input_file, output_file)
    else:
        print(f"File not found: {input_file}")
