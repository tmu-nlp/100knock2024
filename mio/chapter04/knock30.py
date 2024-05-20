#第4章: 形態素解析
#夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，
# その結果をneko.txt.mecabというファイルに保存せよ．このファイルを用いて，以下の問に対応するプログラムを実装せよ．
#なお，問題37, 38, 39はmatplotlibもしくはGnuplotを用いるとよい．

#30. 形態素解析結果の読み込み
#形態素解析結果（neko.txt.mecab）を読み込むプログラムを実装せよ．
# ただし，各形態素は表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）
# をキーとするマッピング型に格納し，
# 1文を形態素（マッピング型）のリストとして表現せよ．第4章の残りの問題では，ここで作ったプログラムを活用せよ．

#用語）マッピング型：辞書型のこと

#準備１：ターミナルでneko.txtをダウンロード
#準備２：ターミナルでneko.txtを形態素解析したものをneko.txt.mecabに保存

#方針１：読み込んだ解析結果をそれぞれ格納する空のlistを2つ作成
#方針２：形態素解析済みのファイルを開き、1行ずつ読み込む
#（解析済みのファイルは形態素ごとに行が分かれていて、一行につき「形態素の表層形,品詞,品詞細分類１,・・・,品詞細分類n,基本形,読み,発音」の形）
#方針３： 改行、タブはpassする（readlines()メソッドを適用しない）
#方針４：l(エル)1の要素が2つなら、l1[1]の要素（＝表層形以外の品詞など）をカンマで区切り、l2に格納
#方針５問題の指示通り4つのキーを指定してdictを作成し、リストsentenceに追加
#方針６：l2の二つ目の要素に句点（。）が来たとき、リストsentence内のdictをresultに追加

#準備作業１
# $ wget https://nlp100.github.io/data/neko.txt 

#準備作業２：コマンド「mecab ./解析したいファイル名 ./解析結果保存先ファイル名」　
#$ mecab ./neko.txt -o ./neko.txt.mecab

#作業１
import re
morph_results = []
sentence = []
 
#作業２
with open("neko.txt.mecab") as f:
  for line in f.readlines():
    if line == "\n" or line[0] == "\t":
            pass
          
#EOSが出たら，辞書のリストsentenceをmorph_resultsに格納
    elif  line == "EOS\n":
            morph_results.append(sentence)
            #次の文の処理に使うため、配列を空にする
            sentence = []

        #re.split()で複数の文字で行を区切り，sentenceに格納
    else:
      #lineをタブまたはカンマで区切る
      #[]で囲むとその中の任意の一文字にマッチ（メリット：一度に複数の文字で区切れる）
      line = re.split("[\t,]",line)
      
            # lineの中身：['一', '名詞', '数', '*', '*', '*', '*', '一', 'イチ', 'イチ\n']
  
      sentence.append({"surface":line[0],"base":line[7],"pos":line[1],"pos1":line[2]})