# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import re
import os
import sys
import time
import numpy as np

import tensorflow as tf
import namespace_utils

import G2S_trainer
import G2S_data_stream
from G2S_model_graph import ModelGraph

from vocab_utils import Vocab



tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL

def map_idx_to_word(predictions, vocab, passage, attn_dist):
    '''
    predictions: [batch, 1]
    '''
    word_size = vocab.vocab_size + 1
    all_words = []
    all_word_idx = []
    for i, idx in enumerate(predictions):
        if idx == vocab.vocab_size:
            k = np.argmax(attn_dist[i,])
            word = batch.instances[i][-1][k]
        else:
            word = vocab.getWord(idx)
        all_words.append(word)
        all_word_idx.append(idx)
    return all_words, all_word_idx

def search(sess, model, vocab, batch, options, decode_mode='greedy'):
    '''
    for greedy search, multinomial search
    '''
    # Run the encoder to get the encoder hidden states and decoder initial state
    (encoder_states, encoder_features, node_idx, node_mask, initial_state) = model.run_encoder(sess, batch, options)
    # encoder_states: [batch_size, passage_len, encode_dim]
    # encoder_features: [batch_size, passage_len, attention_vec_size]
    # node_idx: [batch_size, passage_len]
    # node_mask: [batch_size, passage_len]
    # initial_state: a tupel of [batch_size, gen_dim]

    word_t = batch.sent_inp[:,0]
    state_t = initial_state
    context_t = np.zeros([batch.batch_size, model.encoder_dim])
    coverage_t = np.zeros([batch.batch_size, encoder_states.shape[1]])
    generator_output_idx = [] # store phrase index prediction
    text_results = []
    generator_input_idx = [word_t] # store word index
    for i in xrange(options.max_answer_len):
        if decode_mode == "pointwise": word_t = batch.sent_inp[:,i]
        feed_dict = {}
        feed_dict[model.init_decoder_state] = state_t
        feed_dict[model.context_t_1] = context_t
        feed_dict[model.coverage_t_1] = coverage_t
        feed_dict[model.word_t] = word_t

        feed_dict[model.encoder_states] = encoder_states
        feed_dict[model.encoder_features] = encoder_features
        feed_dict[model.node_idx] = node_idx
        feed_dict[model.node_mask] = node_mask

        if decode_mode in ["greedy", "pointwise", ]:
            prediction = model.greedy_prediction
        elif decode_mode == "multinomial":
            prediction = model.multinomial_prediction

        (state_t, attn_dist_t, context_t, coverage_t, prediction) = sess.run([model.state_t, model.attn_dist_t, model.context_t,
                                                                 model.coverage_t, prediction], feed_dict)
        # convert prediction to word ids
        generator_output_idx.append(prediction)
        prediction = np.reshape(prediction, [prediction.size, 1])
        cur_words, cur_word_idx = map_idx_to_word(prediction) # [batch_size, 1]
        cur_word_idx = np.array(cur_word_idx)
        cur_word_idx = np.reshape(cur_word_idx, [cur_word_idx.size])
        word_t = cur_word_idx
        text_results.append(cur_words)
        generator_input_idx.append(cur_word_idx)

    generator_input_idx = generator_input_idx[:-1] # remove the last word to shift one position to the right
    generator_input_idx = np.stack(generator_input_idx, axis=1) # [batch_size, max_len]
    generator_output_idx = np.stack(generator_output_idx, axis=1) # [batch_size, max_len]

    prediction_lengths = [] # [batch_size]
    sentences = [] # [batch_size]
    for i in xrange(batch.batch_size):
        words = []
        for j in xrange(options.max_answer_len):
            cur_phrase = text_results[j][i]
            words.append(cur_phrase)
            if cur_phrase == "</s>": break # filter out based on end symbol
        prediction_lengths.append(len(words))
        cur_sent = " ".join(words)
        sentences.append(cur_sent)

    return (sentences, prediction_lengths, generator_input_idx, generator_output_idx)

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

    def idx_seq_to_string(self, passage, vocab, options):
        word_size = vocab.vocab_size + 1
        all_words = []
        for i, idx in enumerate(self.tokens):
            cur_word = vocab.getWord(idx)
            if cur_word == 'UNK':
                idx = passage[self.attn_ids[i]]
                cur_word = vocab.getWord(idx)
            all_words.append(cur_word)
        return " ".join(all_words[1:])


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)



def run_beam_search(sess, model, vocab, batch, options):
    # Run encoder
    (encoder_states, encoder_features, node_idx, node_mask, initial_state) = model.run_encoder(sess, batch, options)
    # encoder_states: [1, passage_len, encode_dim]
    # initial_state: a tupel of [1, gen_dim]
    # encoder_features: [1, passage_len, attention_vec_size]
    # node_idx: [1, passage_len]
    # node_mask: [1, passage_len]

    sent_stop_id = vocab.getIndex('</s>')

    # Initialize this first hypothesis
    context_t = np.zeros([model.encoder_dim]) # [encode_dim]
    coverage_t = np.zeros((encoder_states.shape[1])) # [passage_len]
    hyps = []
    hyps.append(Hypothesis([batch.sent_inp[0][0]], [0.0], [-1], initial_state, context_t, coverage_vector=coverage_t))

    # beam search decoding
    results = [] # this will contain finished hypotheses (those that have emitted the </s> token)
    steps = 0
    while steps < options.max_answer_len and len(results) < options.beam_size:
        cur_size = len(hyps) # current number of hypothesis in the beam
        cur_encoder_states = np.tile(encoder_states, (cur_size, 1, 1))
        cur_encoder_features = np.tile(encoder_features, (cur_size, 1, 1)) # [batch_size,passage_len, options.attention_vec_size]
        cur_node_idx = np.tile(node_idx, (cur_size, 1)) # [batch_size, passage_len]
        cur_node_mask = np.tile(node_mask, (cur_size, 1)) # [batch_size, passage_len]
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

        feed_dict[model.encoder_states] = cur_encoder_states
        feed_dict[model.encoder_features] = cur_encoder_features
        feed_dict[model.nodes] = cur_node_idx
        feed_dict[model.nodes_mask] = cur_node_mask
        feed_dict[model.coverage_t_1] = cur_coverage_t_1

        (state_t, context_t, attn_dist_t, coverage_t, topk_log_probs, topk_ids) = sess.run([model.state_t, model.context_t, model.attn_dist_t,
                                                                 model.coverage_t, model.topk_log_probs, model.topk_ids], feed_dict)

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
                # add anony constraint
                #if cur_tok in vocab.anony_ids and cur_tok not in batch.amr_anony_ids:
                #    continue
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
    return hyps_sorted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--out_path', type=str, help='The path to the output file.')
    parser.add_argument('--mode', type=str,default='pointwise', help='The path to the output file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path
    mode = args.mode

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = G2S_trainer.enrich_options(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    edgelabel_vocab = Vocab(model_prefix + ".edgelabel_vocab", fileformat='txt2')
    print('edgelabel_vocab: {}'.format(edgelabel_vocab.word_vecs.shape))
    char_vocab = None
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))


    print('Loading test set from {}.'.format(in_path))
    testset, _, _, _, _ = G2S_data_stream.read_amr_file(in_path)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    batch_size=-1
    if mode not in ('pointwise', 'multinomial', 'greedy', 'greedy_evaluate', ):
        batch_size = 1

    devDataStream = G2S_data_stream.G2SDataStream(testset, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=batch_size)
    print('Number of instances in testDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(devDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, Edgelabel_vocab=edgelabel_vocab,
                                         options=FLAGS, mode="decode")

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
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
        for i in range(total_num):
            cur_batch = devDataStream.get_batch(i)
            if mode == 'pointwise':
                (sentences, prediction_lengths, generator_input_idx,
                 generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    cur_total = cur_batch.answer_lengths[j]
                    cur_correct = 0
                    for k in xrange(cur_total):
                        if generator_output_idx[j,k]== cur_batch.sent_out[j,k]: cur_correct+=1.0
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
                 generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode=mode)
                for j in xrange(cur_batch.batch_size):
                    outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    outfile.write(sentences[j].encode('utf-8') + "\n")
                    outfile.write(str(prediction_lengths[j])+ "\n")
                    outfile.write("========\n")
                outfile.flush()
            elif mode == 'greedy_evaluate':
                print('Batch {}'.format(i))
                (sentences, prediction_lengths, generator_input_idx,
                generator_output_idx) = search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode="greedy")
                for j in xrange(cur_batch.batch_size):
                    ref_outfile.write(cur_batch.instances[j][1].tokText.encode('utf-8') + "\n")
                    pred_outfile.write(sentences[j].encode('utf-8') + "\n")
                ref_outfile.flush()
                pred_outfile.flush()
            elif mode == 'beam_evaluate':
                print('Instance {}'.format(i))
                ref_outfile.write(cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                ref_outfile.flush()
                hyps = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                cur_passage = cur_batch.instances[0][0]
                cur_sent = hyps[0].idx_seq_to_string(cur_passage, word_vocab, FLAGS)
                pred_outfile.write(cur_sent.encode('utf-8') + "\n")
                pred_outfile.flush()
            elif mode == 'beam_search':
                print('Instance {}'.format(i))
                hyps = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                outfile.write("Input: " + cur_batch.instances[0][0].tokText.encode('utf-8') + "\n")
                outfile.write("Truth: " + cur_batch.instances[0][1].tokText.encode('utf-8') + "\n")
                for j in xrange(len(hyps)):
                    hyp = hyps[j]
                    cur_passage = cur_batch.instances[0][0]
                    cur_sent = hyp.idx_seq_to_string(cur_passage, word_vocab, FLAGS)
                    outfile.write("Hyp-{}: ".format(j) + cur_sent.encode('utf-8') + " {}".format(hyp.avg_log_prob()) + "\n")
                outfile.write("========\n")
                outfile.flush()
            else: # beam search
                print('Instance {}'.format(i))
                hyps = run_beam_search(sess, valid_graph, word_vocab, cur_batch, FLAGS)
                outfile.write(cur_batch.id[0] + "\n")
                outfile.write(' '.join(cur_batch.target_ref[0]).encode('utf-8') + "\n")
                for j in xrange(1):
                    hyp = hyps[j]
                    cur_passage = cur_batch.amr_node[0]
                    cur_sent = hyp.idx_seq_to_string(cur_passage, word_vocab, FLAGS)
                    outfile.write(cur_sent.encode('utf-8') + "\n")
                outfile.write("--------\n")
                outfile.write("========\n")
                outfile.flush()
        if mode.endswith('evaluate'):
            ref_outfile.close()
            pred_outfile.close()
        else:
            outfile.close()




