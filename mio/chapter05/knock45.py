#45. 動詞の格パターンの抽出Permalink
#今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい． 動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ． ただし，出力は以下の仕様を満たすようにせよ．

#動詞を含む文節において，最左の動詞の基本形を述語とする
#述語に係る助詞を格とする
#述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
#例）「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」
# →「作り出す」という１つの動詞を含み，
# 「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」であると解析された場合は，次のような出力
#作り出す	で は を
#このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．
#・コーパス中で頻出する述語と格パターンの組み合わせ
#・「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）



import knock41

for sentence in knock41.sentences:  
    for chunk in sentence.chunks:
        for morph1 in chunk.morphs:
            if morph1.pos == "動詞":
                part = [] #動詞にかかる文節の助詞を格納
                for src in chunk.srcs:
                    for morph2 in sentence.chunks[src].morphs:
                        if morph2.pos == "助詞":
                            part.append(morph2.surface)
                if len(part) > 0: ##述語に係る助詞（文節）が複数あるとき
                    sort_part = sorted(list(set(part))) #すべての助詞をスペース区切りで辞書順に並べる
                    part_line = ' '.join(sort_part)
                    print(morph1.base + "\t" + part_line)
                break  #最左だけ処理したらすぐにループを抜ける