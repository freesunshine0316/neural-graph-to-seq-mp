# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np

from vocab_utils import Vocab
import namespace_utils
import NP2P_data_stream
from NP2P_model_graph import ModelGraph

import re

import tensorflow as tf
import NP2P_trainer
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL

def search(sess, model, vocab, batch, options, decode_mode='greedy'):
    '''
    for greedy search, multinomial search
    '''
    # Run the encoder to get the encoder hidden states and decoder initial state
    (phrase_representations, initial_state, encoder_features,phrase_idx, phrase_mask) = model.run_encoder(sess, batch, options)
    # phrase_representations: [batch_size, passage_len, encode_dim]
    # initial_state: a tupel of [batch_size, gen_dim]
    # encoder_features: [batch_size, passage_len, attention_vec_size]
    # phrase_idx: [batch_size, passage_len]
    # phrase_mask: [batch_size, passage_len]

    word_t = batch.gen_input_words[:,0]
    state_t = initial_state
    context_t = np.zeros([batch.batch_size, model.encode_dim])
    coverage_t = np.zeros((batch.batch_size, phrase_representations.shape[1]))
    generator_output_idx = [] # store phrase index prediction
    text_results = []
    generator_input_idx = [word_t] # store word index
    for i in xrange(options.max_answer_len):
        if decode_mode == "pointwise": word_t = batch.gen_input_words[:,i]
        feed_dict = {}
        feed_dict[model.init_decoder_state] = state_t
        feed_dict[model.context_t_1] = context_t
        feed_dict[model.coverage_t_1] = coverage_t
        feed_dict[model.word_t] = word_t

        feed_dict[model.phrase_representations] = phrase_representations
        feed_dict[model.encoder_features] = encoder_features
        feed_dict[model.phrase_idx] = phrase_idx
        feed_dict[model.phrase_mask] = phrase_mask
        if options.with_phrase_projection:
            feed_dict[model.max_phrase_size] = batch.max_phrase_size
            if options.add_first_word_prob_for_phrase:
                feed_dict[model.in_passage_words] = batch.sent1_word
                feed_dict[model.phrase_starts] = batch.phrase_starts



        if decode_mode in ["greedy","pointwise"]:
            prediction = model.greedy_prediction
        elif decode_mode == "multinomial":
            prediction = model.multinomial_prediction

        (state_t, context_t, attn_dist_t, coverage_t, prediction) = sess.run([model.state_t, model.context_t, model.attn_dist_t,
                                                                 model.coverage_t, prediction], feed_dict)
        attn_idx = np.argmax(attn_dist_t, axis=1) # [batch_size]
        # convert prediction to word ids
        generator_output_idx.append(prediction)
        prediction = np.reshape(prediction, [prediction.size, 1])
        [cur_words, cur_word_idx] = batch.map_phrase_idx_to_text(prediction) # [batch_size, 1]
        cur_word_idx = np.array(cur_word_idx)
        cur_word_idx = np.reshape(cur_word_idx, [cur_word_idx.size])
        word_t = cur_word_idx
        cur_words = flatten_words(cur_words) # [batch_size]

        for i, wword in enumerate(cur_words):
            if wword == 'UNK' and attn_idx[i] < len(batch.passage_words[i]):
                cur_words[i] = batch.passage_words[i][attn_idx[i]]

        text_results.append(cur_words)
        generator_input_idx.append(cur_word_idx)

    generator_input_idx = generator_input_idx[:-1] # remove the last word to shift one position to the right
    generator_output_idx = np.stack(generator_output_idx, axis=1) # [batch_size, max_len]
    generator_input_idx = np.stack(generator_input_idx, axis=1) # [batch_size, max_len]

    prediction_lengths = [] # [batch_size]
    sentences = [] # [batch_size]
    for i in xrange(batch.batch_size):
        words = []
        for j in xrange(options.max_answer_len):
            cur_phrase = text_results[j][i]
#             cur_phrase = cur_batch_text[j]
            words.append(cur_phrase)
            if cur_phrase == "</s>": break# filter out based on end symbol
        prediction_lengths.append(len(words))
        cur_sent = " ".join(words)
        sentences.append(cur_sent)

    return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)

def flatten_words(cur_words):
    all_words = []
    for i in xrange(len(cur_words)):
        all_words.append(cur_words[i][0])
    return all_words

class Hypothesis(object):
    def __init__(self, tokens, log_ps, attn, state, context_vector, coverage_vector=None):
        self.tokens = tokens # store all tokens
        self.log_probs = log_ps # store log_probs for each time-step
        self.attn_ids = attn
        self.state = state
        self.context_vector = context_vector
        self.coverage_vector = coverage_vector

    def extend(self, token, log_prob, attn_i, state, context_vector, coverage_vector=None):
        return Hypothesis(self.tokens + [token], self.log_probs + [log_prob], self.attn_ids + [attn_i], state,
                          context_vector, coverage_vector=coverage_vector)

    def latest_token(self):
        return self.tokens[-1]

    def avg_log_prob(self):
        return np.sum(self.log_probs[1:])/ (len(self.tokens)-1)

    def probs2string(self):
        out_string = ""
        for prob in self.log_probs:
            out_string += " %.4f" % prob
        return out_string.strip()

    def idx_seq_to_string(self, passage, id2phrase, vocab, options):
        word_size = vocab.vocab_size + 1
        all_words = []
        for i, idx in enumerate(self.tokens):
            if idx < word_size:
                cur_word = vocab.getWord(idx)
                #if cur_word == 'UNK':
                #    cur_word = passage.words[self.attn_ids[i]]
            else:
                cur_word = id2phrase[idx]
                if not options.withTextChunk:
                    items = re.split('-', cur_word)
                    cur_word = passage.getTokChunk(int(items[0]), int(items[1]))
            all_words.append(cur_word)
        return " ".join(all_words[1:])


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)



def run_beam_search(sess, model, vocab, batch, options):
    # Run encoder
    st = time.time()
    (phrase_representations, initial_state, encoder_features,phrase_idx, phrase_mask) = model.run_encoder(sess, batch, options)
    encoding_dur = time.time() - st
    # phrase_representations: [1, passage_len, encode_dim]
    # initial_state: a tupel of [1, gen_dim]
    # encoder_features: [1, passage_len, attention_vec_size]
    # phrase_idx: [1, passage_len]
    # phrase_mask: [1, passage_len]

    sent_stop_id = vocab.getIndex('</s>')

    # Initialize this first hypothesis
    context_t = np.zeros([model.encode_dim]) # [encode_dim]
    coverage_t = np.zeros((phrase_representations.shape[1])) # [passage_len]
    hyps = []
    hyps.append(Hypothesis([batch.gen_input_words[0][0]], [0.0], [-1], initial_state, context_t, coverage_vector=coverage_t))

    # beam search decoding
    results = [] # this will contain finished hypotheses (those that have emitted the </s> token)
    steps = 0
    while steps < options.max_answer_len and len(results) < options.beam_size:
        cur_size = len(hyps) # current number of hypothesis in the beam
        cur_phrase_representations = np.tile(phrase_representations, (cur_size, 1, 1))
        cur_encoder_features = np.tile(encoder_features, (cur_size, 1, 1)) # [batch_size,passage_len, options.attention_vec_size]
        cur_phrase_idx = np.tile(phrase_idx, (cur_size, 1)) # [batch_size, passage_len]
        cur_phrase_mask = np.tile(phrase_mask, (cur_size, 1)) # [batch_size, passage_len]
        cur_state_t_1 = [] # [2, gen_steps]
        cur_context_t_1 = [] # [batch_size, encoder_dim]
        cur_coverage_t_1 = [] # [batch_size, passage_len]
        cur_word_t = [] # [batch_size]
        for h in hyps:
            cur_state_t_1.append(h.state)
            cur_context_t_1.append(h.context_vector)
            cur_word_t.append(h.latest_token())
            cur_coverage_t_1.append(h.coverage_vector)
        cur_context_t_1 = np.stack(cur_context_t_1, axis=0)
        cur_coverage_t_1 = np.stack(cur_coverage_t_1, axis=0)
        cur_word_t = np.array(cur_word_t)

        cells = [state.c for state in cur_state_t_1]
        hidds = [state.h for state in cur_state_t_1]
        new_c = np.concatenate(cells, axis=0)
        new_h = np.concatenate(hidds, axis=0)
        new_dec_init_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        feed_dict = {}
        feed_dict[model.init_decoder_state] = new_dec_init_state
        feed_dict[model.context_t_1] = cur_context_t_1
        feed_dict[model.word_t] = cur_word_t

        feed_dict[model.phrase_representations] = cur_phrase_representations
        feed_dict[model.encoder_features] = cur_encoder_features
        feed_dict[model.phrase_idx] = cur_phrase_idx
        feed_dict[model.phrase_mask] = cur_phrase_mask
        feed_dict[model.coverage_t_1] = cur_coverage_t_1
        if options.with_phrase_projection:
            feed_dict[model.max_phrase_size] = batch.max_phrase_size
            if options.add_first_word_prob_for_phrase:
                feed_dict[model.in_passage_words] = batch.sent1_word
                feed_dict[model.phrase_starts] = batch.phrase_starts

        (state_t, context_t, attn_dist_t, coverage_t, topk_log_probs, topk_ids) = sess.run([model.state_t, model.context_t,
            model.attn_dist_t, model.coverage_t, model.topk_log_probs, model.topk_ids], feed_dict)

        new_states = [tf.contrib.rnn.LSTMStateTuple(state_t.c[i:i+1, :], state_t.h[i:i+1, :]) for i in xrange(cur_size)]

        # Extend each hypothesis and collect them all in all_hyps
        if steps == 0: cur_size = 1
        all_hyps = []
        for i in xrange(cur_size):
            h = hyps[i]
            cur_state = new_states[i]
            cur_context = context_t[i]
            cur_coverage = coverage_t[i]
            for j in xrange(options.beam_size):
                cur_tok = topk_ids[i, j]
                cur_tok_log_prob = topk_log_probs[i, j]
                cur_attn_i = np.argmax(attn_dist_t[i, :])
                new_hyp = h.extend(cur_tok, cur_tok_log_prob, cur_attn_i, cur_state, cur_context, coverage_vector=cur_coverage)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        # hyps will contain hypotheses for the next step
        hyps = []
        for h in sort_hyps(all_hyps):
            # If this hypothesis is sufficiently long, put in results. Otherwise discard.
            if h.latest_token() == sent_stop_id:
                if steps >= options.min_answer_len:
                    results.append(h)
            # hasn't reached stop token, so continue to extend this hypothesis
            else:
                hyps.append(h)
            if len(hyps) == options.beam_size or len(results) == options.beam_size:
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps
    # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    if len(results)==0:
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted, encoding_dur

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--out_path', type=str, help='The path to the output file.')
    parser.add_argument('--mode', type=str,default='pointwise', help='The path to the output file.')
    parser.add_argument('--beam_size', type=int, default=-1, help='')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path
    mode = args.mode

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = NP2P_trainer.enrich_options(FLAGS)
    if args.beam_size != -1:
        FLAGS.beam_size = args.beam_size

    # load vocabs
    print('Loading vocabs.')
    enc_word_vocab = dec_word_vocab = char_vocab = POS_vocab = NER_vocab = None
    if FLAGS.with_word:
        enc_word_vocab = Vocab(FLAGS.enc_word_vec_path, fileformat='txt2')
        print('enc_word_vocab: {}'.format(enc_word_vocab.word_vecs.shape))
        dec_word_vocab = Vocab(FLAGS.dec_word_vec_path, fileformat='txt2')
        print('dec_word_vocab: {}'.format(dec_word_vocab.word_vecs.shape))
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    if FLAGS.with_POS:
        POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
        print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
    if FLAGS.with_NER:
        NER_vocab = Vocab(model_prefix + ".NER_vocab", fileformat='txt2')
        print('NER_vocab: {}'.format(NER_vocab.word_vecs.shape))


    print('Loading test set.')
    if FLAGS.infile_format == 'fof':
        testset, _ = NP2P_data_stream.read_generation_datasets_from_fof(in_path, isLower=FLAGS.isLower)
    elif FLAGS.infile_format == 'plain':
        testset, _ = NP2P_data_stream.read_all_GenerationDatasets(in_path, isLower=FLAGS.isLower)
    else:
        testset, _ = NP2P_data_stream.read_all_GQA_questions(in_path, isLower=FLAGS.isLower, switch=FLAGS.switch_qa)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    batch_size = -1
    if mode not in ('pointwise', 'multinomial', 'greedy', 'greedy_evaluate', ): batch_size = 1
    devDataStream = NP2P_data_stream.QADataStream(testset, enc_word_vocab, dec_word_vocab, char_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=batch_size)
    print('Number of instances in testDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(devDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(enc_word_vocab=enc_word_vocab, dec_word_vocab=dec_word_vocab, char_vocab=char_vocab,
                        options=FLAGS, mode="decode")

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        total = 0
        correct = 0
        if mode.endswith('evaluate'):
            ref_outfile = open(out_path+ ".ref", 'wt')
            pred_outfile = open(out_path+ ".pred", 'wt')
        else:
            outfile = open(out_path, 'wt')
        total_num = devDataStream.get_num_batch()
        devDataStream.reset()
        total_dur = 0
        for i in range(total_num):
            cur_batch = devDataStream.get_batch(i)
            if mode == 'pointwise':
                (sentences, prediction_lengths, generator_input_idx,
                 generator_output_idx) = search(sess, valid_graph, dec_word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    cur_total = cur_batch.answer_lengths[j]
                    cur_correct = 0
                    for k in xrange(cur_total):
                        if generator_output_idx[j,k]== cur_batch.in_answer_words[j,k]: cur_correct+=1.0
                    total += cur_total
                    correct += cur_correct
                    outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    outfile.write(sentences[j].encode('utf-8') + "\n")
                    outfile.write("========\n")
                outfile.flush()
                print('Current dev accuracy is %d/%d=%.2f' % (correct, total, correct/ float(total) * 100))
            elif mode in ['greedy', 'multinomial']:
                print('Batch {}'.format(i))
                (sentences, prediction_lengths, generator_input_idx,
                 generator_output_idx) = search(sess, valid_graph, dec_word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    outfile.write(cur_batch.instances[j][1].ID_num.encode('utf-8') + "\n")
                    outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    outfile.write(sentences[j].encode('utf-8') + "\n")
                    outfile.write(str(prediction_lengths[j])+ "\n")
                    outfile.write("========\n")
                outfile.flush()
            elif mode == 'greedy_evaluate':
                print('Batch {}'.format(i))
                (sentences, prediction_lengths, generator_input_idx,
                generator_output_idx) = search(sess, valid_graph, dec_word_vocab, cur_batch, FLAGS, decode_mode="greedy")
                for j in xrange(cur_batch.batch_size):
                    ref_outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    pred_outfile.write(sentences[j].encode('utf-8') + "\n")
                ref_outfile.flush()
                pred_outfile.flush()
            elif mode == 'beam_evaluate':
                print('Instance {}'.format(i))
                ref_outfile.write(cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                ref_outfile.flush()
                hyps, _ = run_beam_search(sess, valid_graph, dec_word_vocab, cur_batch, FLAGS)
                cur_passage = cur_batch.instances[0][0]
                cur_id2phrase = None
                if FLAGS.with_phrase_projection: (cur_phrase2id, cur_id2phrase) = cur_batch.phrase_vocabs[0]
                cur_sent = hyps[0].idx_seq_to_string(cur_passage, cur_id2phrase, dec_word_vocab, FLAGS)
                pred_outfile.write(cur_sent.encode('utf-8') + "\n")
                pred_outfile.flush()
            elif mode == 'beam_search':
                print('Instance {}'.format(i))
                hyps, dur = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                total_dur += dur
                outfile.write("Input: " + cur_batch.instances[0][0].tokText.encode('utf-8') + "\n")
                outfile.write("Truth: " + cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                for j in xrange(len(hyps)):
                    hyp = hyps[j]
                    cur_passage = cur_batch.instances[0][0]
                    cur_id2phrase = None
                    if FLAGS.with_phrase_projection: (cur_phrase2id, cur_id2phrase) = cur_batch.phrase_vocabs[0]
                    cur_sent = hyp.idx_seq_to_string(cur_passage, cur_id2phrase, word_vocab, FLAGS)
                    outfile.write("Hyp-{}: ".format(j) + cur_sent.encode('utf-8') + " {}".format(hyp.avg_log_prob()) + "\n")
                outfile.write("========\n")
                outfile.flush()
            else: # beam search
                print('Instance {}'.format(i))
                hyps, dur = run_beam_search(sess, valid_graph, dec_word_vocab, cur_batch, FLAGS)
                total_dur += dur
                outfile.write("None\n")
                outfile.write(cur_batch.target_ref[0].encode('utf-8') + "\n")
                for j in xrange(1):
                    hyp = hyps[j]
                    cur_passage = None #cur_batch.instances[0][0]
                    cur_id2phrase = None
                    if FLAGS.with_phrase_projection: (cur_phrase2id, cur_id2phrase) = cur_batch.phrase_vocabs[0]
                    cur_sent = hyp.idx_seq_to_string(cur_passage, cur_id2phrase, dec_word_vocab, FLAGS)
                    outfile.write(cur_sent.encode('utf-8') + "\n")
                outfile.write("--------\n")
                outfile.write("========\n")
                outfile.flush()
        if mode.endswith('evaluate'):
            ref_outfile.close()
            pred_outfile.close()
        else:
            outfile.close()
        print('Total encoding time {}'.format(total_dur))



