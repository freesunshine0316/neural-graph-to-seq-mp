
import sys
import os

output_dict = {}
ref_dict = {}
for i,line in enumerate(open(sys.argv[1],'rU')):
    if i%5 == 0:
        id = line.strip()
    elif i%5 == 1:
        ref = line.strip().lower().split()
        ref_dict[id] = ref
    elif i%5 == 2:
        rst = line.strip().replace('</s>','',10).split()
        output_dict[id] = rst

dataset_type = None
if sys.argv[1].startswith('test'):
    dataset_type = 'test'
elif sys.argv[1].startswith('dev'):
    dataset_type = 'dev'

fout = open(sys.argv[1]+'.1best','w')
fref = open(sys.argv[1]+'.ref','w')
for id in output_dict.keys():
    print >>fout, ' '.join(output_dict[id]).lower()
    print >>fref, ' '.join(ref_dict[id]).lower()
fout.close()
fref.close()

os.system('/home/lsong10/ws/exp.graph_to_seq/mosesdecoder/scripts/generic/multi-bleu.perl %s.ref < %s.1best' %(sys.argv[1],sys.argv[1]))

