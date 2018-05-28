import json
import re
import numpy as np
import random
import padding_utils
import phrase_lattice_utils

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines


def read_all_GenerationDatasets(inpath, isLower=True):
    with open(inpath) as dataset_file:
        dataset = json.load(dataset_file, encoding='utf-8')
    all_instances = []
    max_answer_len = 0
    for instance in dataset:
        sent1 = instance['amr'].strip()
        sent2 = instance['sent'].strip()
        id = instance['id'] if 'id' in instance else None
        if sent1 == "" or sent2 == "":
            continue
        max_answer_len = max(max_answer_len, len(sent2.split())) # text2 is the sequence to be generated
        all_instances.append((sent1, sent2, id))
    return all_instances, max_answer_len

def read_generation_datasets_from_fof(fofpath, isLower=True):
    all_paths = read_text_file(fofpath)
    all_instances = []
    max_answer_len = 0
    for cur_path in all_paths:
        print(cur_path)
        (cur_instances, cur_max_answer_len) = read_all_GenerationDatasets(cur_path, isLower=isLower)
        print("cur_max_answer_len: %s" % cur_max_answer_len)
        all_instances.extend(cur_instances)
        if max_answer_len<cur_max_answer_len: max_answer_len = cur_max_answer_len
    return all_instances, max_answer_len

def collect_vocabs(all_instances):
    all_words = set()
    for (sent1, sent2, id) in all_instances:
        all_words.update(re.split("\\s+", sent1))
        all_words.update(re.split("\\s+", sent2))
    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars)

class DataStream(object):
    def __init__(self, all_questions, enc_word_vocab=None, dec_word_vocab=None, char_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.options = options
        if batch_size == -1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (sent1, sent2, id) in all_questions:# sent1 is the long passage or article
            sent1_idx = enc_word_vocab.to_index_sequence(sent1)
            sent2_idx = dec_word_vocab.to_index_sequence(sent2)
            oov_rate1 = 1.0*np.sum(x == enc_word_vocab.vocab_size for x in sent1_idx)/len(sent1_idx)
            oov_rate2 = 1.0*np.sum(x == dec_word_vocab.vocab_size for x in sent2_idx)/len(sent2_idx)
            if oov_rate1 > 0.2 or oov_rate2 > 0.2:
                print('!!!!!oov_rate for ENC {} and DEC {}'.format(oov_rate1, oov_rate2))
                print(sent1)
                print(sent2)
                print('==============')
            if options.max_passage_len != -1: sent1_idx = sent1_idx[:options.max_passage_len]
            if options.max_answer_len != -1: sent2_idx = sent2_idx[:options.max_answer_len]
            instances.append((sent1_idx, sent2_idx, sent1, sent2, id))

        all_questions = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_questions = sorted(all_questions, key=lambda xxx: (len(xxx[0]), len(xxx[1])))
        else:
            pass
        self.num_instances = len(all_questions)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_questions = []
            for i in xrange(batch_start, batch_end):
                cur_questions.append(all_questions[i])
            cur_batch = Batch(cur_questions, options, word_vocab=dec_word_vocab, char_vocab=char_vocab)
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

class Batch(object):
    def __init__(self, instances, options, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None):
        self.options = options

        self.batch_size = len(instances)
        self.vocab = word_vocab

        self.id = [inst[-1] for inst in instances]
        self.source = [inst[-3] for inst in instances]
        self.target_ref = [inst[-2] for inst in instances]

        # create length
        self.sent1_length = [] # [batch_size]
        self.sent2_length = [] # [batch_size]
        for (sent1_idx, sent2_idx, _, _, _) in instances:
            self.sent1_length.append(len(sent1_idx))
            self.sent2_length.append(min(len(sent2_idx)+1, options.max_answer_len))
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)

        # create word representation
        start_id = word_vocab.getIndex('<s>')
        end_id = word_vocab.getIndex('</s>')
        if options.with_word:
            self.sent1_word = [] # [batch_size, sent1_len]
            self.sent2_word = [] # [batch_size, sent2_len]
            self.sent2_input_word = []
            for (sent1_idx, sent2_idx, _, _, _) in instances:
                self.sent1_word.append(sent1_idx)
                self.sent2_word.append(sent2_idx+[end_id])
                self.sent2_input_word.append([start_id]+sent2_idx)
            self.sent1_word = padding_utils.pad_2d_vals(self.sent1_word, len(instances), np.max(self.sent1_length))
            self.sent2_word = padding_utils.pad_2d_vals(self.sent2_word, len(instances), options.max_answer_len)
            self.sent2_input_word = padding_utils.pad_2d_vals(self.sent2_input_word, len(instances), options.max_answer_len)

            self.in_answer_words = self.sent2_word
            self.gen_input_words = self.sent2_input_word
            self.answer_lengths = self.sent2_length

        if options.with_char:
            self.sent1_char = [] # [batch_size, sent1_len]
            self.sent1_char_lengths = []
            for (_, _, sent1, sent2, _) in instances:
                sent1_char_idx = char_vocab.to_character_matrix_for_list(sent1.split()[:options.max_passage_len])
                self.sent1_char.append(sent1_char_idx)
                self.sent1_char_lengths.append([len(x) for x in sent1_char_idx])
            self.sent1_char = padding_utils.pad_3d_vals_no_size(self.sent1_char)
            self.sent1_char_lengths = padding_utils.pad_2d_vals_no_size(self.sent1_char_lengths)


if __name__ == "__main__":
    all_instances, _ = read_all_GenerationDatasets('./data/training.json', True)
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent1.get_length() > 200))
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent2.get_length() > 100))
    print('DONE!')
    all_instances, _ = read_all_GenerationDatasets('./data/test.json', True)
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent1.get_length() > 200))
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent2.get_length() > 100))
    print('DONE!')
