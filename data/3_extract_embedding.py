
import numpy
import sys, os
from collections import Counter

vocab = set(line.strip() for line in open(sys.argv[1], 'rU'))
print 'len(vocab)', len(vocab)

intersect = set()
f = open(sys.argv[2], 'w')
#for line in open('/home/lsong10/ws/data.embedding/glove.840B.300d.txt', 'rU'):
#    word = line.strip().split()[0]
#    if word in vocab:
#        intersect.add(word)
#        print >>f, line.strip()
print len(intersect)

for w in vocab - intersect:
    embedding = ' '.join([str('%.6f'%x) for x in numpy.random.normal(size=600)])
    print >>f, w, embedding

f.close()
