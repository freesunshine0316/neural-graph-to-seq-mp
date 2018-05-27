from graphviz import Digraph
import numpy as np

class prefix_tree_node(object):
    def __init__(self, node_id):
        self.node_id = node_id 
        self.phrase_id = -1# phrase_id==-1 for all intermediate nodes
        self.children = {}
        self.parent = None

    def add_child(self, word, next_node):
        self.children[word] = next_node
        next_node.parent = (word, self)

    def set_phrase_id(self, phrase_id):
        self.phrase_id = phrase_id
    
    def find_child(self, word):
        if self.children.has_key(word): 
            return self.children.get(word)
        else: 
            return None

class prefix_tree(object):
    def __init__(self, phrase2id):
        self.root_node = prefix_tree_node(0)
        self.all_nodes = [self.root_node]
        self.phrase_id_node = {}

        for phrase,phrase_id in dict.iteritems(phrase2id):
            words = phrase.split()
            if len(words)<=1: continue
            cur_node = self.root_node
            for word in words:
                next_node = cur_node.find_child(word)
                if next_node == None:
                    next_node = prefix_tree_node(len(self.all_nodes))
                    cur_node.add_child(word, next_node)
                    self.all_nodes.append(next_node)
                cur_node = next_node
            cur_node.set_phrase_id(phrase_id)
            self.phrase_id_node[phrase_id] = cur_node
    
    def get_phrase_id(self, phrase):
        words = phrase.split()
        cur_node = self.root_node
        for word in words:
            cur_node = cur_node.find_child(word)
            if cur_node is None: return None
        return cur_node.phrase_id
    
    def get_phrase(self, phrase_id):
        if not self.phrase_id_node.has_key(phrase_id): return None
        cur_node = self.phrase_id_node[phrase_id]
        words = []
        while cur_node:
            if cur_node.parent is None: break
            (cur_word, cur_parent) = cur_node.parent
            words.insert(0, cur_word)
            cur_node = cur_parent
        return " ".join(words)
    
    def has_phrase_id(self, phrase_id):
        return self.phrase_id_node.has_key(phrase_id)


    def init_bak(self, phrase_id):
        self.root_node = prefix_tree_node(0)
        self.all_nodes = [self.root_node]

        for phrase, phrase_id in phrase_id.iteritems():
            words = phrase.split()
            cur_node = self.root_node
            for word in words:
                next_node = cur_node.find_child(word)
                if next_node == None:
                    next_node = prefix_tree_node(len(self.all_nodes))
                    cur_node.add_child(word, next_node)
                    self.all_nodes.append(next_node)
                cur_node = next_node
            cur_node.set_phrase_id(phrase_id)

    def __str__(self):
        dot = Digraph(name='prefix_tree')
        for cur_node in self.all_nodes:
            dot.node(str(cur_node.node_id), str(cur_node.phrase_id))
            for edge in cur_node.children.keys():
                cur_edge_node = cur_node.children[edge]
                dot.edge(str(cur_node.node_id), str(cur_edge_node.node_id), edge)

        dot.body.append(r'label = "prefix tree"')
        return dot.source


class lattice_node(object):
    def __init__(self, node_id):
        self.node_id = node_id
        self.out_deges = []

    def add_edge(self, edge):
        self.out_deges.append(edge)
    
    def get_edge(self, i):
        if len(self.out_deges)<i+1: return None
        return self.out_deges[i]
    
    def __str__(self):
        out_string = "node_id: {}\n".format(self.node_id)
        for i, cur_edge in enumerate(self.out_deges):
            out_string += "edge {}: {}\n".format(i, cur_edge)
        return out_string
    


class lattice_edge(object):
    def __init__(self, phrase, phrase_id, tail_node):
        self.phrase = phrase
        self.phrase_id = phrase_id
        self.tail_node = tail_node

    def __str__(self):
        out_string = "phrase: {}, phrase id: {}, tail node: {}".format(self.phrase, self.phrase_id, self.tail_node.node_id)
        return out_string

class phrase_lattice(object):
    def __init__(self, toks, word_vocab=None, prefix_tree=None):
        '''
        create a lattice for toks 
        '''
        # create all single word edge
        self.all_nodes = []
        self.start_node = lattice_node(-1)
        self.all_nodes.append(self.start_node)
        prev_node = self.start_node
        for i, tok in enumerate(toks):
            next_node = lattice_node(i)
            self.all_nodes.append(next_node)
            cur_tok_id = 0
            if word_vocab is not None: cur_tok_id = word_vocab.getIndex(tok)
            cur_edge = lattice_edge(tok, cur_tok_id, next_node)
            prev_node.add_edge(cur_edge)
            prev_node = next_node
        
        if prefix_tree is None: return

        # match all possible phrases
        for cur_start_node in self.all_nodes:
            cur_node = cur_start_node
            cur_prefix_node = prefix_tree.root_node
            while cur_node and cur_prefix_node:
                cur_edge = cur_node.get_edge(0) # the first edge is single word edge
                if cur_edge is None: break
                cur_word = cur_edge.phrase
                cur_tail_node = cur_edge.tail_node
                cur_prefix_node = cur_prefix_node.find_child(cur_word)
                if cur_prefix_node is None: break
                if cur_prefix_node.phrase_id>-1 and cur_tail_node.node_id-cur_start_node.node_id>1: # one phrase is match
                    # add one edge
                    cur_phrase_id = cur_prefix_node.phrase_id
#                     cur_phrase = id2phrase[cur_phrase_id]
                    cur_phrase = prefix_tree.get_phrase(cur_phrase_id)
                    new_edge = lattice_edge(cur_phrase, cur_phrase_id, cur_tail_node)
                    cur_start_node.add_edge(new_edge)
                cur_node = cur_tail_node

    def sample_a_partition(self, max_matching=False):
        phrases = []
        phrase_ids = []
        cur_node = self.start_node
        while cur_node:
            all_edges = cur_node.out_deges
            edge_size = len(all_edges)
            if edge_size<=0: break
            if max_matching:
                max_len = -1
                sampled_idx = 0
                for i, cur_edge in enumerate(all_edges):
                    cur_length = len(cur_edge.phrase.split())
                    if max_len<cur_length: 
                        max_len=cur_length
                        sampled_idx = i
            else:
                sampled_idx = np.random.randint(edge_size,size=1)[0]
            sampled_edge = all_edges[sampled_idx]
            phrases.append(sampled_edge.phrase)
            phrase_ids.append(sampled_edge.phrase_id)
            cur_node = sampled_edge.tail_node
        return (phrases, phrase_ids)
    
    def __str__(self):
        dot = Digraph(name='phrase_lattice')
        for cur_node in self.all_nodes:
            dot.node(str(cur_node.node_id), str(cur_node.node_id))
            for cur_edge in cur_node.out_deges:
                tail_node = cur_edge.tail_node
                cur_phrase = cur_edge.phrase
                dot.edge(str(cur_node.node_id), str(tail_node.node_id), cur_phrase)
        dot.body.append(r'label = "phrase lattice"')
        return dot.source

def collect_all_possible_phrases(sentence, max_chunk_len):
    phrase2id = {}
    id2phrase = {}
    words = sentence.split()
    for i in xrange(len(words)):
        for j in xrange(i, i+max_chunk_len):
            if j>=len(words): break
            cur_phrase = " ".join(words[i:j+1])
            if phrase2id.has_key(cur_phrase): continue
            cur_index = len(phrase2id)
            phrase2id[cur_phrase] = cur_index
            id2phrase[cur_index] = cur_phrase
    return (phrase2id, id2phrase)


if __name__ == "__main__":
    ''' # test create prefix tree
    phrase_id = {"a": 0, "a b f": 1, "a b c":2, "a b d":3, "b":4, "b d":5, "b e":6}
    tree = prefix_tree(phrase_id)
    print(tree)
    #'''

    ''' # test creating lattice
    sentence = "a b cc dd ee f g hhh iiiii jj kk lm"
    toks = sentence.split()
    lattice = phrase_lattice(toks)
    print(lattice)
    #'''
    
    src_sentence = "what is the significance of the periodic table ?"
    tgt_sentence = "what is a periodic table ?"
    
    # collect phrases from src_setence
    max_chunk_len = 4
    (phrase2id, id2phrase) = collect_all_possible_phrases(src_sentence, max_chunk_len=max_chunk_len)
    
    # create prefix tree
    tree = prefix_tree(phrase2id)
#     print(tree)
    #'''
    for phrase in phrase2id.keys():
        phrase_id = tree.get_phrase_id(phrase)
        cur_phrase = tree.get_phrase(phrase_id)
        print(phrase)
        print(phrase_id)
        print(cur_phrase)
        print()
    #'''
    
    '''
    # create lattice for the target sentence
    lattice = phrase_lattice(tgt_sentence.split(), word_vocab=None, prefix_tree=tree)
#     print(lattice)

    # sample partitions
    (phrases, phrase_ids) = lattice.sample_a_partition()
    print(phrases)
    print(phrase_ids)

    (phrases, phrase_ids) = lattice.sample_a_partition()
    print(phrases)
    print(phrase_ids)

    (phrases, phrase_ids) = lattice.sample_a_partition()
    print(phrases)
    print(phrase_ids)

    (phrases, phrase_ids) = lattice.sample_a_partition(max_matching=True)
    print(phrases)
    print(phrase_ids)
    #'''