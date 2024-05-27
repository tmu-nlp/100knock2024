import graphviz
from knock41 import get_chanks_by_sentence

sentences = get_chanks_by_sentence()
dot = graphviz.Digraph(format='png')

dot.attr('node', fontname = 'Noto Sans CJK JP')
dot.attr('edge', fontname = 'Noto Sans CJK JP')

target = sentences[3]

for chank in target:
  dot.node(f"{chank.srcs}", chank.marge_morphs())

for chank in target:
  if chank.dst != -1:
    dot.edge(f"{chank.srcs}", f"{chank.dst}")

dot.render('./kotsugi/chapter05/knock44', format='png')
