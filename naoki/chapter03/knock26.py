import re
inf_dic2 = {}
#.items()でキーと値を同時に扱えるようにしている
for key, text in inf_dic.items():
  #テキスト内の//と'が2回以上5回以下連続する部分を空文字列で置換
  inf_dic2[key] = re.sub(r'(\\\'){2,5}' , '', text)
inf_dic2