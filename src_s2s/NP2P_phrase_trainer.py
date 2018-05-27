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
import NP2P_beam_decoder
import metric_utils
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
cc = SmoothingFunction()

FLAGS = None
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


def evaluate(sess, valid_graph, devDataStream, word_vocab, options=None):
    devDataStream.reset()
    gen = []
    ref = []
    for batch_index in xrange(devDataStream.get_num_batch()): # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        (sentences, _, _, _) = NP2P_beam_decoder.search(sess, valid_graph, word_vocab, cur_batch, FLAGS, decode_mode="greedy")
        for i in xrange(cur_batch.batch_size):
            cur_ref = cur_batch.instances[i][1].tokText
            cur_prediction = sentences[i]
            ref.append([cur_ref.split()])
            gen.append(cur_prediction.split())
    return corpus_bleu(ref, gen, smoothing_function=cc.method3)

def main(_):
    print('Configurations:')
    print(FLAGS)

    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/NP2P.{}".format(FLAGS.suffix)
    init_model_prefix = FLAGS.init_model # "/u/zhigwang/zhigwang1/sentence_generation/mscoco/logs/NP2P.phrase_ce_train"
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    print('Loading train set.')
    if FLAGS.infile_format == 'fof':
        trainset, train_ans_len = NP2P_data_stream.read_generation_datasets_from_fof(FLAGS.train_path, isLower=FLAGS.isLower)
        if FLAGS.max_answer_len>train_ans_len: FLAGS.max_answer_len = train_ans_len
    else: 
        trainset, train_ans_len = NP2P_data_stream.read_all_GQA_questions(FLAGS.train_path, isLower=FLAGS.isLower)
    print('Number of training samples: {}'.format(len(trainset)))

    print('Loading test set.')
    if FLAGS.infile_format == 'fof':
        testset, test_ans_len = NP2P_data_stream.read_generation_datasets_from_fof(FLAGS.test_path, isLower=FLAGS.isLower)
    else:
        testset, test_ans_len = NP2P_data_stream.read_all_GQA_questions(FLAGS.test_path, isLower=FLAGS.isLower)
    print('Number of test samples: {}'.format(len(testset)))

    max_actual_len = max(train_ans_len, test_ans_len)
    print('Max answer length: {}, truncated to {}'.format(max_actual_len, FLAGS.max_answer_len))

    word_vocab = None
    POS_vocab = None
    NER_vocab = None
    char_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + ".best.model"
    if os.path.exists(init_model_prefix + ".best.model.index"):
        has_pretrained_model = True
        print('!!Existing pretrained model. Loading vocabs.')
        if FLAGS.with_word:
            word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
            print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        if FLAGS.with_char:
            char_vocab = Vocab(init_model_prefix + ".char_vocab", fileformat='txt2')
            print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
        if FLAGS.with_POS:
            POS_vocab = Vocab(init_model_prefix + ".POS_vocab", fileformat='txt2')
            print('POS_vocab: {}'.format(POS_vocab.word_vecs.shape))
        if FLAGS.with_NER:
            NER_vocab = Vocab(init_model_prefix + ".NER_vocab", fileformat='txt2')
            print('NER_vocab: {}'.format(NER_vocab.word_vecs.shape))
    else:
        print('Collecting vocabs.')
        (allWords, allChars, allPOSs, allNERs) = NP2P_data_stream.collect_vocabs(trainset)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of allChars: {}'.format(len(allChars)))
        print('Number of allPOSs: {}'.format(len(allPOSs)))
        print('Number of allNERs: {}'.format(len(allNERs)))

        if FLAGS.with_word:
            word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        if FLAGS.with_char:
            char_vocab = Vocab(voc=allChars, dim=FLAGS.char_dim, fileformat='build')
            char_vocab.dump_to_txt2(path_prefix + ".char_vocab")
        if FLAGS.with_POS:
            POS_vocab = Vocab(voc=allPOSs, dim=FLAGS.POS_dim, fileformat='build')
            POS_vocab.dump_to_txt2(path_prefix + ".POS_vocab")
        if FLAGS.with_NER:
            NER_vocab = Vocab(voc=allNERs, dim=FLAGS.NER_dim, fileformat='build')
            NER_vocab.dump_to_txt2(path_prefix + ".NER_vocab")

    print('word vocab size {}'.format(word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build DataStream ... ')
    trainDataStream = NP2P_data_stream.QADataStream(trainset, word_vocab, char_vocab, POS_vocab, NER_vocab, options=FLAGS,
                 isShuffle=True, isLoop=True, isSort=True)

    devDataStream = NP2P_data_stream.QADataStream(testset, word_vocab, char_vocab, POS_vocab, NER_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    sys.stdout.flush()

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, options=FLAGS, mode="rl_train_for_phrase")

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab,
                                         NER_vocab=NER_vocab, options=FLAGS, mode="decode")

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        if has_pretrained_model:
            print("Restoring model from " + init_model_prefix + ".best.model")
            saver.restore(sess, init_model_prefix + ".best.model")
            print("DONE!")
        sys.stdout.flush()

        # for first-time rl training, we get the current BLEU score
        print("First-time rl training, get the current BLEU score on dev")
        sys.stdout.flush()
        best_bleu = evaluate(sess, valid_graph, devDataStream, word_vocab, options=FLAGS)
        print('First-time bleu = %.4f' % best_bleu)
        log_file.write('First-time bleu = %.4f\n' % best_bleu)

        print('Start the training loop.')
        sys.stdout.flush()
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            cur_batch = trainDataStream.nextBatch()
            if FLAGS.with_baseline:
                # greedy search
                (greedy_sentences, _, _, _) = NP2P_beam_decoder.search(sess, valid_graph, word_vocab, 
                                                                           cur_batch, FLAGS, decode_mode="greedy")

            if FLAGS.with_target_lattice:
                (sampled_sentences, sampled_prediction_lengths, sampled_generator_input_idx, 
                    sampled_generator_output_idx) = cur_batch.sample_a_partition()
            else:
                # multinomial sampling
                (sampled_sentences, sampled_prediction_lengths, sampled_generator_input_idx, 
                    sampled_generator_output_idx) = NP2P_beam_decoder.search(sess, valid_graph, word_vocab, 
                                                                           cur_batch, FLAGS, decode_mode="multinomial")
            # calculate rewards
            rewards = []
            for i in xrange(cur_batch.batch_size):
#                 print(sampled_sentences[i])
#                 print(sampled_generator_input_idx[i])
#                 print(sampled_generator_output_idx[i])
                cur_toks = cur_batch.instances[i][1].tokText.split()
#                 r = sentence_bleu([cur_toks], sampled_sentences[i].split(), smoothing_function=cc.method3)
                r = 1.0
                b = 0.0
                if FLAGS.with_baseline:
                    b = sentence_bleu([cur_toks], greedy_sentences[i].split(), smoothing_function=cc.method3)
#                 r = metric_utils.evaluate_captions([cur_toks],[sampled_sentences[i]])
#                 b = metric_utils.evaluate_captions([cur_toks],[greedy_sentences[i]])
                rewards.append(1.0* (r-b))
            rewards = np.array(rewards, dtype=np.float32)
#             sys.exit(-1)

            # update parameters
            feed_dict = train_graph.run_encoder(sess, cur_batch, FLAGS, only_feed_dict=True)
            feed_dict[train_graph.reward] = rewards
            feed_dict[train_graph.gen_input_words] = sampled_generator_input_idx
            feed_dict[train_graph.in_answer_words] = sampled_generator_output_idx
            feed_dict[train_graph.answer_lengths] = sampled_prediction_lengths
            (_, loss_value) = sess.run([train_graph.train_op, train_graph.loss], feed_dict)
            total_loss += loss_value

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                log_file.flush()
                sys.stdout.flush()
                total_loss = 0.0

                # Evaluate against the validation set.
                start_time = time.time()
                print('Validation Data Eval:')
                dev_bleu = evaluate(sess, valid_graph, devDataStream, word_vocab, options=FLAGS)
                print('Dev bleu = %.4f' % dev_bleu)
                log_file.write('Dev bleu = %.4f\n' % dev_bleu)
                log_file.flush()
                if best_bleu < dev_bleu:
                    print('Saving weights, BLEU {} (prev_best) < {} (cur)'.format(best_bleu, dev_bleu))
                    best_bleu = dev_bleu
                    saver.save(sess, best_path) # TODO: save model
                duration = time.time() - start_time
                print('Duration %.3f sec' % (duration))
                sys.stdout.flush()
                log_file.write('Duration %.3f sec\n' % (duration))
                log_file.flush()

    log_file.close()

def enrich_options(options):
    if not options.__dict__.has_key("infile_format"):
        options.__dict__["infile_format"] = "old"

    if not options.__dict__.has_key("with_target_lattice"):
        options.__dict__["with_target_lattice"] = False

    if not options.__dict__.has_key("with_baseline"):
        options.__dict__["with_baseline"] = True

    if not options.__dict__.has_key("add_first_word_prob_for_phrase"):
        options.__dict__["add_first_word_prob_for_phrase"] = False

    if not options.__dict__.has_key("pretrain_with_max_matching"):
        options.__dict__["pretrain_with_max_matching"] = False
    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    FLAGS = enrich_options(FLAGS)

    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
