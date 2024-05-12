import argparse

parser = argparse.ArgumentParser()   
parser.add_argument('X')
parser.add_argument('Y')
parser.add_argument('Z')

args = parser.parse_args()

def text_prt(x,y,z):
    print(x,"時の",y,"は",z)

if __name__ == "__main__":
    text_prt(args.X,args.Y,args.Z)