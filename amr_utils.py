
def read_anonymized(amr_lst, amr_node, amr_edge):
    assert sum(x=='(' for x in amr_lst) == sum(x==')' for x in amr_lst)
    cur_str = amr_lst[0]
    cur_id = len(amr_node)
    amr_node.append(cur_str)

    i = 1
    while i < len(amr_lst):
        if amr_lst[i].startswith(':') == False: ## cur cur-num_0
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, ':value'))
            i = i + 1
        elif amr_lst[i].startswith(':') and len(amr_lst) == 2: ## cur :edge
            nxt_str = 'num_unk'
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 1
        elif amr_lst[i].startswith(':') and amr_lst[i+1] != '(': ## cur :edge nxt
            nxt_str = amr_lst[i+1]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 2
        elif amr_lst[i].startswith(':') and amr_lst[i+1] == '(': ## cur :edge ( ... )
            number = 1
            j = i+2
            while j < len(amr_lst):
                number += (amr_lst[j] == '(')
                number -= (amr_lst[j] == ')')
                if number == 0:
                    break
                j += 1
            assert number == 0 and amr_lst[j] == ')', ' '.join(amr_lst[i+2:j])
            nxt_id = read_anonymized(amr_lst[i+2:j], amr_node, amr_edge)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = j + 1
        else:
            assert False, ' '.join(amr_lst)
    return cur_id

if __name__ == '__main__':
    for path in ['data/dev-dfs-linear_src.txt', 'data/test-dfs-linear_src.txt', 'data/training-dfs-linear_src.txt', ]:
        print path
        for i, line in enumerate(open(path, 'rU')):
            amr_node = []
            amr_edge = []
            read_anonymized(line.strip().split(), amr_node, amr_edge)
