
import json
import sys
import cPickle
import re
from collections import Counter
import codecs

def update(l, v):
    v.update([x.lower() for x in l])

def update_vocab(path, vocab, vocab_edge, vocab_node):
    words = []
    words_edge = []
    words_node = []
    data = json.load(open(path,'rU'))
    for inst in data:
        words += inst['sent'].lower().strip().split()
        words += [x for x in inst['amr'].lower().strip().split() if x[0] != ':']
        words_edge += [x for x in inst['amr'].lower().strip().split() if x[0] == ':']
        words_node += [x for x in inst['amr'].lower().strip().split() if re.search('_[0-9]+', x) != None or x == 'num_unk']
    update(words, vocab)
    update(words_edge, vocab_edge)
    update(words_node, vocab_node)

def output(d, path):
    f = codecs.open(path,'w',encoding='utf-8')
    for k,v in sorted(d.items(), key=lambda x:-x[1]):
        print >>f, k
    f.close()

##################

vocab = Counter()
vocab_edge = Counter()
vocab_node = Counter()
update_vocab('training.json', vocab, vocab_edge, vocab_node)
print len(vocab), len(vocab_edge), len(vocab_node)

output(vocab, 'vocab.txt')
output(vocab_edge, 'vocab_edge.txt')
output(vocab_node, 'vocab_node.txt')

