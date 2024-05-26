str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = str.split()
pick_first_list = [ 1, 5, 6, 7, 8, 9, 15, 16, 19 ]
pick_idx = 0
elements = {}

for i in range(len(words)):
  value = ""

  if ((i + 1) in pick_first_list):
    value = words[i][:1]
  else:
    value = words[i][:2]
  
  elements[i+1] = value

print(elements)
