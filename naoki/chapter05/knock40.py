class Morph(object):
    def __init__(self, pos):
        self.surface = pos[0]
        self.base = pos[7]
        self.pos = pos[1]
        self.pos1 = pos[2]

with open("ai.ja.txt.parsed", "r") as f:
    lines = f.readlines()
    ai_list = []
    morph_list = []
    for text in lines:
        if text[0:3]=="EOS":
            if ai_list:
                morph_list.append(ai_list)
                ai_list = []
            continue
        if text[0]=="*":
            continue
        pos = text.split("\t")
        temp = pos[1].split(",")
        pos.pop()
        pos.extend(temp)
        ai_list.append(Morph(pos).__dict__)
print(morph_list)