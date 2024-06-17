import urllib.request
with urllib.request.urlopen( "https://nlp100.github.io/data/popular-names.txt") as f:
    content = f.read().decode("utf-8")
    #タブ:\t
    print(content.replace('\t',' '))
"""
$ cat 'popular-names.txt' | sed 's/\t/ /'
$ cat 'popular-names.txt' | tr '\t' ' '
$ cat 'popular-names.txt' | expand
"""
