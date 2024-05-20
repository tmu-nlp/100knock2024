#NO34(名詞の連接)
import MeCab
#マッピング型は辞書型のことを指す
#表層形（文章中に使用されている単語のこと）
#基本形(終止形、原型)
#MeCab出力　表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
result = []
sentence = []
with open("neko.txt.mecab") as f:
  for line in f:
    #(表層形)と（品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音）に分ける(_tでの分離)
    l1 = line.split("\t")
    #うまく分離できればlen(l1)==2となる。
    if len(l1) == 2:
      #（品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音）を「,」で区切っていく
      l2 = l1[1].split(",")
      #list out of rangeを防ぐ(エラーが起きたときは、基本形に表層形を利用する)
      try:
        #l1は表層形のみを含む　l2は[0:品詞 1:品詞細分類1・・・のようになっている]
        sentence.append({"surface": l1[0], "base": l2[7], "pos": l2[0], "pos1": l2[1]})
      except IndexError:
        sentence.append({"surface": l1[0], "base": l1[0], "pos": l2[0], "pos1": l2[1]})


      #句点「。」がきたら辞書を作成する
      if l2[1] == "句点":
        result.append(sentence)
        #追加したら、sentenceを空にする
        sentence = []
noun_list = []
#resultはネスト(listのなかにdictという多重構造)になっているため、2回for文を回す必用がある。
#knock30の結果をみると、ネスト構造の意味が理解しやすい(resultという一番大きなリストからsentenseというリストを取り出し、sentenseのリストの中に入っているdictを取り出す。)
for sentense in result:
    for dic in sentense:
       count = 0
       sent = ""
       for i in range(len(sentense)):
          #名詞が見つかるごとに、countを1ずつ増やしていく
          if sentense[i]["pos"] == "名詞":
            count += 1
            #sentで文字列をつなぎ合わせていく
            sent += sentense[i]["surface"]
       else:
            #もし名詞以外のものがきて、countが2以上(なにかしら文字がつなぎ合っている）であればsentをnoun_listに追加する
            if count >= 2:
              noun_list.append(sent)
            #countとsentをリセットする
            count = 0
            sent = ""
noun_set = set(noun_list)
print(noun_set)