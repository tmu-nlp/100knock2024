import argparse
import random

parser = argparse.ArgumentParser()   
parser.add_argument('text')

args = parser.parse_args()

def randomized(text):
    splited = text.split()
    shuffled = []

    for elm in splited:
        if len(elm) <= 4:
            shuffled.append(elm)
        else:
            middle = list(elm[1:-1])
            random.shuffle(middle)
            shuffled_elm = elm[0] + ''.join(middle) + elm[-1]
            shuffled.append(shuffled_elm)

    return ' '.join(shuffled)


if __name__=='__main__':
    print(randomized(args.text))