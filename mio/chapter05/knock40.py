#40. 係り受け解析結果の読み込み（形態素
#形態素を表すクラスMorphを実装せよ．
# このクラスは表層形（surface），基本形（base），品詞（pos），品詞細分類1（pos1）をメンバ変数に持つこととする．
# さらに，係り受け解析の結果（ai.ja.txt.parsed）を読み込み，
# 各文をMorphオブジェクトのリストとして表現し，冒頭の説明文の形態素列を表示せよ．


#態素解析結果の文字列（morphemes）を受け取り、メンバ変数surface、base、pos、pos1 を設定
class Morph:
    def __init__(self, morpheme):
        
        #➀morphemeをタブの前後で分割し、前半部分をsurface/後半部分をattributeに格納
        #morphemeの例）人工	名詞,一般,*,*,*,*,人工,ジンコウ,ジンコー
        surface, attribute = morpheme.split('\t')
        attribute_list  = attribute.split(",")
        self.surface = surface
        self.base = attribute_list[6] #人工
        self.pos = attribute_list[0] #名詞
        self.pos1 = attribute_list[1] #一般


sentences_list = []
morphemes_list =[]   

with open("ai.ja.txt.parsed", "r") as input_text:
    for line in input_text:
        
        #係受け関係を表す行はよまない（例）* 0 17D 1/1 -1.776924
        if line[0] =="*":
            continue 
        
        elif line =="EOS\n":
            #morphemes_listに形態素が含まれていれば、sentences_listにmorphemes_listを追加
            if len(morphemes_list) >0:
                sentences_list.append(morphemes_list)
            morphemes_list = []
        
        #lineをMorphクラスのインスタンスにして、morphemes_listに追加    
        else:
            morphemes_list.append(Morph(line))
            

for morph in sentences_list[0]:
  #vars()：オブジェクト morph の属性を辞書形式で返す
  print(vars(morph))
 
    
        