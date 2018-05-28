import json
import re
import numpy as np
import random
import padding_utils
import amr_utils

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines

def read_amr_file(inpath):
    nodes = [] # [batch, node_num,]
    in_neigh_indices = [] # [batch, node_num, neighbor_num,]
    in_neigh_edges = []
    out_neigh_indices = [] # [batch, node_num, neighbor_num,]
    out_neigh_edges = []
    sentences = [] # [batch, sent_length,]
    ids = []
    max_in_neigh = 0
    max_out_neigh = 0
    max_node = 0
    max_sent = 0
    with open(inpath, "rU") as f:
        for inst in json.load(f):
            amr = inst['amr']
            sent = inst['sent'].strip().split()
            id = inst['id'] if inst.has_key('id') else None
            amr_node = []
            amr_edge = []
            amr_utils.read_anonymized(amr.strip().split(), amr_node, amr_edge)
            # 1.
            nodes.append(amr_node)
            # 2. & 3.
            in_indices = [[i,] for i, x in enumerate(amr_node)]
            in_edges = [[':self',] for i, x in enumerate(amr_node)]
            out_indices = [[i,] for i, x in enumerate(amr_node)]
            out_edges = [[':self',] for i, x in enumerate(amr_node)]
            for (i,j,lb) in amr_edge:
                in_indices[j].append(i)
                in_edges[j].append(lb)
                out_indices[i].append(j)
                out_edges[i].append(lb)
            in_neigh_indices.append(in_indices)
            in_neigh_edges.append(in_edges)
            out_neigh_indices.append(out_indices)
            out_neigh_edges.append(out_edges)
            # 4.
            sentences.append(sent)
            ids.append(id)
            # update lengths
            max_in_neigh = max(max_in_neigh, max(len(x) for x in in_indices))
            max_out_neigh = max(max_out_neigh, max(len(x) for x in out_indices))
            max_node = max(max_node, len(amr_node))
            max_sent = max(max_sent, len(sent))
    return zip(nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, sentences, ids), \
            max_node, max_in_neigh, max_out_neigh, max_sent

def read_amr_from_fof(fofpath):
    all_paths = read_text_file(fofpath)
    all_instances = []
    max_node = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_sent = 0
    for cur_path in all_paths:
        print(cur_path)
        cur_instances, cur_node, cur_in_neigh, cur_out_neigh, cur_sent = read_amr_file(cur_path)
        all_instances.extend(cur_instances)
        max_node = max(max_node, cur_node)
        max_in_neigh = max(max_in_neigh, cur_in_neigh)
        max_out_neigh = max(max_out_neigh, cur_out_neigh)
        max_sent = max(max_sent, cur_sent)
    return all_instances, max_node, max_in_neigh, max_out_neigh, max_sent

def collect_vocabs(all_instances):
    all_words = set()
    all_chars = set()
    all_edgelabels = set()
    # nodes: [corpus_size,node_num,]
    # neigh_indices & neigh_edges: [corpus_size,node_num,neigh_num,]
    # sentence: [corpus_size,sent_len,]
    for (nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, sentence, id) in all_instances:
        all_words.update(nodes)
        all_words.update(sentence)
        for edges in in_neigh_edges:
            all_edgelabels.update(edges)
        for edges in out_neigh_edges:
            all_edgelabels.update(edges)
    for w in all_words:
        all_chars.update(w)
    return (all_words, all_chars, all_edgelabels)

class G2SDataStream(object):
    def __init__(self, all_instances, word_vocab=None, char_vocab=None, edgelabel_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.options = options
        if batch_size ==-1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, sentence, id) in all_instances:
            if options.max_node_num != -1 and len(nodes) > options.max_node_num:
                continue # remove very long passages
            in_neigh_indices = [x[:options.max_in_neigh_num] for x in in_neigh_indices]
            in_neigh_edges = [x[:options.max_in_neigh_num] for x in in_neigh_edges]
            out_neigh_indices = [x[:options.max_out_neigh_num] for x in out_neigh_indices]
            out_neigh_edges = [x[:options.max_out_neigh_num] for x in out_neigh_edges]

            nodes_idx = word_vocab.to_index_sequence_for_list(nodes)
            nodes_chars_idx = None
            if options.with_char:
                nodes_chars_idx = char_vocab.to_character_matrix_for_list(nodes, max_char_per_word=options.max_char_per_word)
            in_neigh_edges_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in in_neigh_edges]
            out_neigh_edges_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in out_neigh_edges]
            sentence_idx = word_vocab.to_index_sequence_for_list(sentence[:options.max_answer_len])
            instances.append((nodes_idx, nodes_chars_idx,
                in_neigh_indices, in_neigh_edges_idx, out_neigh_indices, out_neigh_edges_idx, sentence_idx, sentence, id))

        all_instances = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda inst: (len(inst[0]), len(inst[-2])))

        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = G2SBatch(cur_instances, options, word_vocab=word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class G2SBatch(object):
    def __init__(self, instances, options, word_vocab=None):
        self.options = options

        self.amr_node = [x[0] for x in instances]
        self.id = [x[-1] for x in instances]
        self.target_ref = [x[-2] for x in instances] # list of tuples
        self.batch_size = len(instances)
        self.vocab = word_vocab

        # create length
        self.node_num = [] # [batch_size]
        self.sent_len = [] # [batch_size]
        for (nodes_idx, nodes_chars_idx, in_neigh_indices, in_neigh_edges_idx, out_neigh_indices, out_neigh_edges_idx,
                sentence_idx, sentence, id) in instances:
            self.node_num.append(len(nodes_idx))
            self.sent_len.append(min(len(sentence_idx)+1, options.max_answer_len))
        self.node_num = np.array(self.node_num, dtype=np.int32)
        self.sent_len = np.array(self.sent_len, dtype=np.int32)

        # node char num
        if options.with_char:
            self.nodes_chars_num = [[len(nodes_chars_idx) for nodes_chars_idx in instance[1]] for instance in instances]
            self.nodes_chars_num = padding_utils.pad_2d_vals_no_size(self.nodes_chars_num)

        # neigh mask
        self.in_neigh_mask = [] # [batch_size, node_num, neigh_num]
        self.out_neigh_mask = []
        for instance in instances:
            ins = []
            for in_neighs in instance[2]:
                ins.append([1 for _ in in_neighs])
            self.in_neigh_mask.append(ins)
            outs = []
            for out_neighs in instance[4]:
                outs.append([1 for _ in out_neighs])
            self.out_neigh_mask.append(outs)
        self.in_neigh_mask = padding_utils.pad_3d_vals_no_size(self.in_neigh_mask)
        self.out_neigh_mask = padding_utils.pad_3d_vals_no_size(self.out_neigh_mask)

        # create word representation
        start_id = word_vocab.getIndex('<s>')
        end_id = word_vocab.getIndex('</s>')

        self.nodes = [x[0] for x in instances]
        if options.with_char:
            self.nodes_chars = [inst[1] for inst in instances] # [batch_size, sent_len, char_num]
        self.in_neigh_indices = [x[2] for x in instances]
        self.in_neigh_edges = [x[3] for x in instances]
        self.out_neigh_indices = [x[4] for x in instances]
        self.out_neigh_edges = [x[5] for x in instances]

        self.sent_inp = []
        self.sent_out = []
        for _, _, _, _, _, _, sentence_idx, sentence, id in instances:
            if len(sentence_idx) < options.max_answer_len:
                self.sent_inp.append([start_id,]+sentence_idx)
                self.sent_out.append(sentence_idx+[end_id,])
            else:
                self.sent_inp.append([start_id,]+sentence_idx[:-1])
                self.sent_out.append(sentence_idx)

        # making ndarray
        self.nodes = padding_utils.pad_2d_vals_no_size(self.nodes)
        if options.with_char:
            self.nodes_chars = padding_utils.pad_3d_vals_no_size(self.nodes_chars)
        self.in_neigh_indices = padding_utils.pad_3d_vals_no_size(self.in_neigh_indices)
        self.in_neigh_edges = padding_utils.pad_3d_vals_no_size(self.in_neigh_edges)
        self.out_neigh_indices = padding_utils.pad_3d_vals_no_size(self.out_neigh_indices)
        self.out_neigh_edges = padding_utils.pad_3d_vals_no_size(self.out_neigh_edges)

        assert self.in_neigh_mask.shape == self.in_neigh_indices.shape
        assert self.in_neigh_mask.shape == self.in_neigh_edges.shape
        assert self.out_neigh_mask.shape == self.out_neigh_indices.shape
        assert self.out_neigh_mask.shape == self.out_neigh_edges.shape

        # [batch_size, sent_len_max]
        self.sent_inp = padding_utils.pad_2d_vals(self.sent_inp, len(self.sent_inp), options.max_answer_len)
        self.sent_out = padding_utils.pad_2d_vals(self.sent_out, len(self.sent_out), options.max_answer_len)


if __name__ == "__main__":
    print('testset')
    all_instances, max_node_num, max_in_neigh_num, max_out_neigh_num, max_sent_len = read_amr_file('./data/test.json')
    print(max_in_neigh_num)
    print(max_out_neigh_num)
    print(max_node_num)
    print(max_sent_len)
    print('DONE!')

