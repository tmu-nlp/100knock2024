import argparse

parser = argparse.ArgumentParser()   
parser.add_argument('code')

args = parser.parse_args()

def cipher(code):
    modified = ''
    for letter in code:
        if 'a' <= letter <= 'z':
            modified += chr(219 - ord(letter))
        else:
            modified += letter

    return modified

if __name__ == '__main__':
    modified = cipher(args.code)
    print(modified)