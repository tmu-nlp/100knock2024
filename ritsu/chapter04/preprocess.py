import MeCab

def parse_text(input_file, output_file):
    mecab = MeCab.Tagger()
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parsed_line = mecab.parse(line)
            outfile.write(parsed_line)

if __name__ == "__main__":
    parse_text('neko.txt', 'neko.txt.mecab')
