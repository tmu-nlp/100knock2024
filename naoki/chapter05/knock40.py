'''
こんな感じの文章が入っている
'EOS\n', '* 0 7D 0/1 -1.062288\n', 'これ\t名詞,代名詞,一般,*,*,*,これ,コレ,コレ\n', 'は\t助詞,係助詞,*,*,*,*,は,ハ,ワ\n', '* 1 2D 2/3 4.950724\n', '「\t記号,括弧開,*,*,*,*,「,「,「\n', '研究\t名詞,サ変接続,*,*,*,*,研究,ケンキュウ,ケンキュー\n', '分野\t名詞,一般,*,*,*,*,分野,ブンヤ,ブンヤ\n', 'の\t助詞,連体化,*,*,*,*,の,ノ,ノ\n', '* 2 7D 2/5 -1.062288\n', '細分\t名詞,サ変接続,*,*,*,*,細分,サイブン,サイブン\n', '化\t名詞,接尾,サ変接続,*,*,*,化,カ,カ\n', 'そのもの\t名詞,一般,*,*,*,*,そのもの,ソノモノ,ソノモノ\n', '」\t記号,括弧閉,*,*,*,*,」,」,」\n', 'で\t助動詞,*,*,*,特殊・ダ,連用形,だ,デ,デ\n', 'あり\t助動詞,*,*,*,五段・ラ行アル,連用形,ある,アリ,アリ\n', '、\t記号,読点,*,*,*,*,、,、,、\n', '* 3 4D 1/2 3.535311\n', '「\t記号,括弧開,*,*,*,*,「,「,「\n', '立派\t名詞,形容動詞語幹,*,*,*,*,立派,リッパ,リッパ\n', 
'''
class Morph():
    def __init__(self, pos):
        self.surface = pos[0]
        self.base = pos[7]
        self.pos = pos[1]
        self.pos1 = pos[2]
with open("ai.ja.txt.parsed", "r") as f:
    lines = f.readlines()
    list = []
    morph_list = []
    for text in lines:
        if text[0:3]=="EOS":
            #listを空にする
            if list:
                morph_list.append(list)
                list = []
            continue
        if text[0]=="*":
            continue
        pos = text.split("\t")
        #例えば、pos[1]には名詞,一般,*,*,*,*,人工,ジンコウ,ジンコーが入る
        temp = pos[1].split(",")
        pos.pop() #\nを削除
        pos.extend(temp) #ココがわからない
        list.append(Morph(pos).__dict__)
#print(morph_list)
print(pos)
'''
今は{'surface': 'の', 'base': 'の', 'pos': '名詞', 'pos1': '非自立'}, {'surface': 'は', 'base': 'は', 'pos': '助詞','pos1': '係助詞'}のようなものがテキスト
'''