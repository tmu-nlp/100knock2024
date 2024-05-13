import re
inf_dic3 = {}
for key, text in inf_dic2.items():
  pattern = "(?<=\}\}\<br \/\>（)\[{2}"#<br />（フランス語]]:[[Dieu et mon droit|神と我が権利]]）
  text = re.sub(pattern, '', text)

  pattern = "\[{2}.*?\|.*?px\|(?=.*?\]\])"#'[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]',
  text = re.sub(pattern, '', text)

  pattern = "(?<=(\|))\[{2}"
  text = re.sub(pattern, '', text)

  pattern = "(?<=\}{2}（)\[{2}"#スコットランド・ゲール語]]）\n*{{lang|cy
  text = re.sub(pattern, '', text)

  pattern = "(?<=\>（)\[{2}.*?\|"#[[グレートブリテン及びアイルランド連合王国]]成立<br />（1800年合同法]]）
  text = re.sub(pattern, '', text)

  pattern = "(?<=（.{4}).*?\[{2}.*?\)\|" #'[[イングランド王国]]／[[スコットランド王国]]<br />（両国とも[[合同法 (1707年)|1707年合同法]]まで）',
  text = re.sub(pattern, '', text)

  pattern = "\[{2}.*?\|"#[[(除去)|]]の処理
  text = re.sub(pattern, '', text)

  pattern = "(\[{2}|\]{2})"#最後に残ったやつを処理
  inf_dic3[key] = re.sub(pattern, '', text)
inf_dic3