
import sys

ids = []
id_dict = {}
amrs = []
amr_str = ''
for line in open(sys.argv[1],'rU'):
    if line.startswith('#'):
        if line.startswith('# ::id'):
            id = line.lower().strip().split()[2]
            ids.append(id)
            id_dict[id] = len(ids)-1
        continue
    line = line.strip()
    if line == '':
        if amr_str != '':
            amrs.append(amr_str.strip())
            amr_str = ''
    else:
        amr_str = amr_str + line + ' '

if amr_str != '':
    amrs.append(amr_str.strip())
    amr_str = ''

if len(sys.argv) == 3:
    for line in open(sys.argv[2],'rU'):
        id = line.lower().strip()
        print amrs[id_dict[id]]

