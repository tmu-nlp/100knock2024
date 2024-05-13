def cipher(string):
  result = ""
  for char in string:
    if (char.islower()):
      result += chr(219 - ord(char))
    else:
      result += char
  return result

print(cipher("I am Kotsugi"))
