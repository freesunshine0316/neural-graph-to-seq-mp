import json
import re
import numpy as np
import random
import padding_utils
from sent_utils import QASentence
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
        if sent1 == "" or sent2 == "":
            continue
        max_answer_len = max(max_answer_len, len(sent2.split())) # text2 is the sequence to be generated
        all_instances.append((sent1, sent2))
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
    for (sent1, sent2) in all_instances:
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
        for (sent1, sent2) in all_questions:# sent1 is the long passage or article
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
            instances.append((sent1_idx, sent2_idx, sent2))

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
            cur_batch = QAQuestionBatch(cur_questions, options, word_vocab=dec_word_vocab, char_vocab=char_vocab)
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

class QAQuestionBatch(object):
    def __init__(self, instances, options, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None):
        self.options = options

        self.batch_size = len(instances)
        self.vocab = word_vocab

        self.source = None
        self.target_ref = [inst[2] for inst in instances]

        # create length
        self.sent1_length = [] # [batch_size]
        self.sent2_length = [] # [batch_size]
        for (sent1, sent2, _) in instances:
            self.sent1_length.append(len(sent1))
            self.sent2_length.append(min(len(sent2)+1, options.max_answer_len))
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)

        # create word representation
        start_id = word_vocab.getIndex('<s>')
        end_id = word_vocab.getIndex('</s>')
        if options.with_word:
            self.sent1_word = [] # [batch_size, sent1_len]
            self.sent2_word = [] # [batch_size, sent2_len]
            self.sent2_input_word = []
            for (sent1, sent2, _) in instances:
                self.sent1_word.append(sent1)
                self.sent2_word.append(sent2+[end_id])
                self.sent2_input_word.append([start_id]+sent2)
            self.sent1_word = padding_utils.pad_2d_vals(self.sent1_word, len(self.sent1_word), options.max_passage_len)
            self.sent2_word = padding_utils.pad_2d_vals(self.sent2_word, len(self.sent2_word), options.max_answer_len)
            self.sent2_input_word = padding_utils.pad_2d_vals(self.sent2_input_word, len(self.sent2_input_word), options.max_answer_len)

            self.in_answer_words = self.sent2_word
            self.gen_input_words = self.sent2_input_word
            self.answer_lengths = self.sent2_length


    def build_phrase_vocabs(self):
        self.phrase_vocabs = []
        word_size = self.vocab.vocab_size + 1

        self.phrase_starts = []
        self.phrase_ends = []
        self.phrase_idx = []
        self.phrase_lengths = []
        self.max_phrase_size = 0
        if self.options.with_target_lattice:
            self.target_lattices = []
        for (sent1, sent2, sent3) in self.instances:
            # collect all phrases
            if self.options.withSyntaxChunk:
                (cur_phrase_starts, cur_phrase_ends, _) = sent1.collect_all_syntax_chunks(self.options.max_chunk_len)
            else:
                (cur_phrase_starts, cur_phrase_ends) = sent1.collect_all_possible_chunks(self.options.max_chunk_len)

            # collect phrase vocab and map phrase into phrase_id
            cur_phrase2id = {}
            cur_phrase_idx = []
            for i in xrange(len(cur_phrase_starts)):
                cur_start = cur_phrase_starts[i]
                cur_end = cur_phrase_ends[i]
                cur_phrase = sent1.getTokChunk(cur_start, cur_end)
                cur_index = None
                if cur_start==cur_end:
                    cur_index = self.vocab.getIndex(cur_phrase)
                elif cur_phrase2id.has_key(cur_phrase):
                    cur_index = cur_phrase2id[cur_phrase]
                else:
                    cur_index = len(cur_phrase2id) + word_size
                    cur_phrase2id[cur_phrase] = cur_index
                cur_phrase_idx.append(cur_index)
            cur_phrase_vocab = phrase_lattice_utils.prefix_tree(cur_phrase2id)
            self.phrase_vocabs.append(cur_phrase_vocab)
            self.phrase_starts.append(cur_phrase_starts)
            self.phrase_ends.append(cur_phrase_ends)
            self.phrase_idx.append(cur_phrase_idx)
            self.phrase_lengths.append(len(cur_phrase_starts))
            cur_phrase_size = len(cur_phrase2id)
            if self.max_phrase_size<cur_phrase_size: self.max_phrase_size = cur_phrase_size

            if self.options.with_target_lattice:
                cur_lattice = phrase_lattice_utils.phrase_lattice(sent2.words, word_vocab=self.vocab, prefix_tree=cur_phrase_vocab)
                self.target_lattices.append(cur_lattice)

        self.phrase_starts = padding_utils.pad_2d_vals_no_size(self.phrase_starts) # [batch_size, phrase_size]
        self.phrase_ends = padding_utils.pad_2d_vals_no_size(self.phrase_ends) # [batch_size, phrase_size]
        self.phrase_idx = padding_utils.pad_2d_vals_no_size(self.phrase_idx) # [batch_size, phrase_size]
        self.phrase_lengths = np.array(self.phrase_lengths, dtype=np.int32) # [batch_size]

    def map_phrase_idx_to_text(self, samples):
        '''
        sample: [batch_size, length] of idx
        '''
        word_size = self.vocab.vocab_size + 1
        all_words = []
        all_word_idx = []
        for i in xrange(len(samples)):
#             cur_passage = self.instances[i][0]
            cur_sample = samples[i]
            if self.options.with_phrase_projection: cur_phrase_vocab = self.phrase_vocabs[i]
            cur_words = []
            cur_word_idx = []
            for idx in cur_sample:
                if idx<word_size:
                    cur_word = self.vocab.getWord(idx)
                elif not cur_phrase_vocab.has_phrase_id(idx): # if an OOV phrase is sampled, reset it to UNK
                    idx = self.vocab.vocab_size
                    cur_word = self.vocab.getWord(idx)
                else:
#                     if not cur_id2phrase.has_key(idx):
#                         print(cur_id2phrase)
#                         print(idx)
#                     cur_word = cur_id2phrase[idx]
                    cur_word = cur_phrase_vocab.get_phrase(idx)
#                     if not self.options.withTextChunk:
#                         items = re.split('-', cur_word)
#                         cur_word = cur_passage.getTokChunk(int(items[0]), int(items[1]))
                    idx = self.vocab.getIndex(re.split("\\s+", cur_word)[-1]) # take the last word of a phrase as the input word for decoding
                cur_words.append(cur_word)
                cur_word_idx.append(idx)
            all_words.append(cur_words)
            all_word_idx.append(cur_word_idx)
        return (all_words, all_word_idx) # [batch_size, length]

    def sample_a_partition(self, max_matching=False):
        word_size = self.vocab.vocab_size + 1
        sentences = []
        prediction_lengths = []
        generator_input_idx = []
        generator_output_idx = []
        for i, cur_lattice in enumerate(self.target_lattices):
            (cur_phrases, cur_phrase_ids) = cur_lattice.sample_a_partition(max_matching=max_matching)
            sentences.append(" ".join(cur_phrases))
            prediction_lengths.append(len(cur_phrases))
            generator_output_idx.append(cur_phrase_ids)
            cur_input_idx = [self.gen_input_words[i][0]]
            for cur_phrase, cur_phrase_id in zip(cur_phrases, cur_phrase_ids):
                if cur_phrase_id<word_size:
                    cur_word_id = cur_phrase_id
                elif not self.phrase_vocabs[i].has_phrase_id(cur_phrase_id): # if an OOV phrase is sampled, reset it to UNK
                    cur_word_id = self.vocab.vocab_size
                else:
                    cur_word_id = self.vocab.getIndex(re.split("\\s+", cur_phrase)[-1]) # take the last word of a phrase as the input word for decoding
                cur_input_idx.append(cur_word_id)
            generator_input_idx.append(cur_input_idx[:-1])

        generator_input_idx = padding_utils.pad_2d_vals(generator_input_idx, len(generator_input_idx), self.options.max_answer_len)
        generator_output_idx = padding_utils.pad_2d_vals(generator_output_idx, len(generator_output_idx), self.options.max_answer_len)
        return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)



if __name__ == "__main__":
    all_instances, _ = read_all_GenerationDatasets('./data/training.json', True)
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent1.get_length() > 200))
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent2.get_length() > 100))
    print('DONE!')
    all_instances, _ = read_all_GenerationDatasets('./data/test.json', True)
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent1.get_length() > 200))
    print(1.0*sum(1 for sent1, sent2, sent3 in all_instances if sent2.get_length() > 100))
    print('DONE!')
