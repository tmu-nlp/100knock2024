inf_dic4 = {}
for key, text in inf_dic3.items():
  pattern = '\{\{.*?\{\{center\|' #{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}',
  text = re.sub(pattern, '', text)

  pattern = '\{\{.*?\|.*?\|.{2}\|'
  text = re.sub(pattern, '', text) #'{{仮リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}'

  pattern = '\<ref.*?\>.*?\<\/ref\>'#<ref>で囲んでいるもの
  text = re.sub(pattern, '', text)

  pattern = '\<ref.*?\>|\<br \/\>'#<ref>単体+<br />単体
  text = re.sub(pattern, '', text)

  pattern = '\{\{lang\|.*?\|'#公式国名、標語の外国語タイトルを消すため
  text = re.sub(pattern, '', text)

  pattern = '\{\{.*?\}\}'#確立年月日2,3,4の{を除去する
  text = re.sub(pattern, '', text)

  pattern = '\}\}'# 公式国名、標語、国家、他元首等氏名2の}を除去する
  text = re.sub(pattern, '', text)

  inf_dic4[key] = text
inf_dic4