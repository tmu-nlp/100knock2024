import random
text = list(input().split())

for words in text:
    if len(words) > 4:
        head = words[0]
        tail = words[len(words)-1]

        body = list(words[1:len(words)-1])
        random.shuffle(body)
        body = "".join(body)

        words = head + body + tail
    
    print(words,end=" ")
print("")