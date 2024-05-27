import re
pattern = "\[\[Category:(.*?)(?:\|.*?|)\]\]"
#()は指定の箇所のみ抽出で、2つ目は1つ目の条件を満たす者のうちいらないものを削っている。(?:x)で非キャプチャグル―プ \|で|の意味を消す効果がある。なので|x or 無を非キャプチャにしている
result = re.findall(pattern, UK_text)
result