# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import codecs

from vocab_utils import Vocab
import namespace_utils
import NP2P_data_stream
from NP2P_model_graph import ModelGraph

FLAGS = None
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
cc = SmoothingFunction()

import metric_utils

import platform
def get_machine_name():
    return platform.node()

def vec2string(val):
    result = ""
    for v in val:
        result += " {}".format(v)
    return result.strip()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def document_bleu(vocab, gen, ref, suffix=''):
    genlex = [vocab.getLexical(x)[1] for x in gen]
    reflex = [[vocab.getLexical(x)[1],] for x in ref]
    #return metric_utils.evaluate_captions(genlex,reflex)
    genlst = [x.split() for x in genlex]
    reflst = [[x[0].split()] for x in reflex]
    f = codecs.open('gen.txt'+suffix,'w','utf-8')
    for line in genlex:
        print(line, end='\n', file=f)
    f.close()
    f = codecs.open('ref.txt'+suffix,'w','utf-8')
    for line in reflex:
        print(line[0], end='\n', file=f)
    f.close()
    return corpus_bleu(reflst, genlst, smoothing_function=cc.method3)


def evaluate(sess, valid_graph, devDataStream, options=None, suffix=''):
    devDataStream.reset()
    gen = []
    ref = []
    dev_loss = 0.0
    dev_right = 0.0
    dev_total = 0.0
    for batch_index in xrange(devDataStream.get_num_batch()): # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        if valid_graph.mode == 'evaluate':
            accu_value, loss_value = valid_graph.run_ce_training(sess, cur_batch, options, only_eval=True)
            dev_loss += loss_value
            dev_right += accu_value
            dev_total += np.sum(cur_batch.answer_lengths)
        elif valid_graph.mode == 'evaluate_bleu':
            gen.extend(valid_graph.run_greedy(sess, cur_batch, options).tolist())
            ref.extend(cur_batch.in_answer_words.tolist())
        else:
            assert False

    if valid_graph.mode == 'evaluate':
        return {'dev_loss':dev_loss, 'dev_accu':1.0*dev_right/dev_total, 'dev_right':dev_right, 'dev_total':dev_total, }
    else:
        return {'dev_bleu':document_bleu(valid_graph.dec_word_vocab,gen,ref,suffix), }



def main(_):
    print('Configurations:')
    print(FLAGS)

    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/NP2P.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    print('Loading training set.')
    trainset, train_ans_len = NP2P_data_stream.read_all_GenerationDatasets(FLAGS.train_path, isLower=FLAGS.isLower)
    print('Number of training samples: {}'.format(len(trainset)))

    print('Loading dev set.')
    devset, dev_ans_len = NP2P_data_stream.read_all_GenerationDatasets(FLAGS.test_path, isLower=FLAGS.isLower)
    print('Number of dev samples: {}'.format(len(devset)))

    if FLAGS.finetune_path != "":
        print('Loading finetune set.')
        ftset, ft_ans_len = NP2P_data_stream.read_all_GenerationDatasets(FLAGS.ft_path, isLower=FLAGS.isLower)
        print('Number of finetune samples: {}'.format(len(ftset)))
    else:
        ftset, ft_ans_len = (None, 0)

    max_actual_len = max(train_ans_len, ft_ans_len, dev_ans_len)
    print('Max answer length: {}, truncated to {}'.format(max_actual_len, FLAGS.max_answer_len))

    enc_word_vocab = None
    dec_word_vocab = None
    char_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + ".best.model"
    if os.path.exists(best_path + ".index"):
        has_pretrained_model = True
        print('!!Existing pretrained model. Loading vocabs.')
        if FLAGS.with_word:
            enc_word_vocab = Vocab(FLAGS.enc_word_vec_path, fileformat='txt2')
            dec_word_vocab = Vocab(FLAGS.dec_word_vec_path, fileformat='txt2')
            print('Encoder word vocab: {}'.format(enc_word_vocab.word_vecs.shape))
            print('Decoder word vocab: {}'.format(dec_word_vocab.word_vecs.shape))
        if FLAGS.with_char:
            char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
            print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    else:
        print('Collecting vocabs.')
        (allWords, allChars) = NP2P_data_stream.collect_vocabs(trainset)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of allChars: {}'.format(len(allChars)))

        if FLAGS.with_word:
            enc_word_vocab = Vocab(FLAGS.enc_word_vec_path, fileformat='txt2')
            dec_word_vocab = Vocab(FLAGS.dec_word_vec_path, fileformat='txt2')
        if FLAGS.with_char:
            char_vocab = Vocab(voc=allChars, dim=FLAGS.char_dim, fileformat='build')
            char_vocab.dump_to_txt2(path_prefix + ".char_vocab")

    print('Encoder word vocab size {}'.format(enc_word_vocab.vocab_size))
    print('Decoder word vocab size {}'.format(dec_word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build DataStream ... ')
    trainDataStream = NP2P_data_stream.DataStream(trainset, enc_word_vocab, dec_word_vocab, char_vocab, options=FLAGS,
                 isShuffle=True, isLoop=True, isSort=True)
    devDataStream = NP2P_data_stream.DataStream(devset, enc_word_vocab, dec_word_vocab, char_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    if ftset != None:
        ftDataStream = NP2P_data_stream.DataStream(ftset, enc_word_vocab, dec_word_vocab, char_vocab, options=FLAGS,
                     isShuffle=True, isLoop=True, isSort=True)
        print('Number of instances in ftDataStream: {}'.format(ftDataStream.get_num_instance()))
        print('Number of batches in ftDataStream: {}'.format(ftDataStream.get_num_batch()))

    sys.stdout.flush()

    init_scale = 0.01
    # initialize the best bleu and accu scores for current training session
    best_accu = FLAGS.best_accu if FLAGS.__dict__.has_key('best_accu') else 0.0
    best_bleu = FLAGS.best_bleu if FLAGS.__dict__.has_key('best_bleu') else 0.0
    if best_accu > 0.0:
        print('With initial dev accuracy {}'.format(best_accu))
    if best_bleu > 0.0:
        print('With initial dev BLEU score {}'.format(best_bleu))

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(enc_word_vocab=enc_word_vocab, dec_word_vocab=dec_word_vocab, char_vocab=char_vocab,
                        POS_vocab=None, NER_vocab=None, options=FLAGS, mode=FLAGS.mode)

        assert FLAGS.mode in ('ce_train', 'rl_train', )
        valid_mode = 'evaluate' if FLAGS.mode == 'ce_train' else 'evaluate_bleu'

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(enc_word_vocab=enc_word_vocab, dec_word_vocab=dec_word_vocab, char_vocab=char_vocab,
                        POS_vocab=None, NER_vocab=None, options=FLAGS, mode=valid_mode)

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            print(var)
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        if has_pretrained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

            if FLAGS.mode == 'rl_train' and abs(best_bleu) < 0.00001:
                print("Getting BLEU score for the model")
                best_bleu = evaluate(sess, valid_graph, devDataStream, options=FLAGS)['dev_bleu']
                FLAGS.best_bleu = best_bleu
                namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                print('BLEU = %.4f' % best_bleu)
                log_file.write('BLEU = %.4f\n' % best_bleu)
            if FLAGS.mode == 'ce_train' and abs(best_accu) < 0.00001:
                print("Getting ACCU score for the model")
                best_accu = evaluate(sess, valid_graph, devDataStream, options=FLAGS)['dev_accu']
                FLAGS.best_accu = best_accu
                namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                print('ACCU = %.4f' % best_accu)
                log_file.write('ACCU = %.4f\n' % best_accu)

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            cur_batch = trainDataStream.nextBatch()
            if FLAGS.mode == 'rl_train':
                loss_value = train_graph.run_rl_training_2(sess, cur_batch, FLAGS)
            elif FLAGS.mode == 'ce_train':
                loss_value = train_graph.run_ce_training(sess, cur_batch, FLAGS)
            total_loss += loss_value

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()


            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps or \
                     (trainDataStream.get_num_batch() > 10000 and (step + 1) % 2000 == 0):
                print()
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                log_file.flush()
                sys.stdout.flush()
                total_loss = 0.0

                if ftset != None:
                    best_accu, best_bleu = fine_tune(sess, saver, FLAGS, log_file,
                            ftDataStream, devDataStream, train_graph, valid_graph, path_prefix, best_accu, best_bleu)
                else:
                    best_accu, best_bleu = validate_and_save(sess, saver, FLAGS, log_file,
                            devDataStream, valid_graph, path_prefix, best_accu, best_bleu)
                start_time = time.time()

    log_file.close()

def validate_and_save(sess, saver, FLAGS, log_file,
    devDataStream, valid_graph, path_prefix, best_accu, best_bleu):
    best_path = path_prefix + ".best.model"
    # Evaluate against the validation set.
    start_time = time.time()
    print('Validation Data Eval:')
    res_dict = evaluate(sess, valid_graph, devDataStream, options=FLAGS)
    if valid_graph.mode == 'evaluate':
        dev_loss = res_dict['dev_loss']
        dev_accu = res_dict['dev_accu']
        dev_right = int(res_dict['dev_right'])
        dev_total = int(res_dict['dev_total'])
        print('Dev loss = %.4f' % dev_loss)
        log_file.write('Dev loss = %.4f\n' % dev_loss)
        print('Dev accu = %.4f %d/%d' % (dev_accu, dev_right, dev_total))
        log_file.write('Dev accu = %.4f %d/%d\n' % (dev_accu, dev_right, dev_total))
        log_file.flush()
        if best_accu < dev_accu:
            print('Saving weights, ACCU {} (prev_best) < {} (cur)'.format(best_accu, dev_accu))
            saver.save(sess, best_path)
            best_accu = dev_accu
            FLAGS.best_accu = dev_accu
            namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    else:
        dev_bleu = res_dict['dev_bleu']
        print('Dev bleu = %.4f' % dev_bleu)
        log_file.write('Dev bleu = %.4f\n' % dev_bleu)
        log_file.flush()
        if best_bleu < dev_bleu:
            print('Saving weights, BLEU {} (prev_best) < {} (cur)'.format(best_bleu, dev_bleu))
            saver.save(sess, best_path)
            best_bleu = dev_bleu
            FLAGS.best_bleu = dev_bleu
            namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    duration = time.time() - start_time
    print('Duration %.3f sec' % (duration))
    sys.stdout.flush()

    log_file.write('Duration %.3f sec\n' % (duration))
    log_file.flush()
    return best_accu, best_bleu


def fine_tune(sess, saver, FLAGS, log_file,
    ftDataStream, devDataStream, train_graph, valid_graph, path_prefix, best_accu, best_bleu):
    print('=====Start the fine tuning.')
    sys.stdout.flush()
    train_size = ftDataStream.get_num_batch()
    max_steps = train_size * 3
    best_path = path_prefix + ".best.model"
    total_loss = 0.0
    start_time = time.time()
    for step in xrange(max_steps):
        cur_batch = ftDataStream.nextBatch()
        if FLAGS.mode == 'rl_train':
            loss_value = train_graph.run_rl_training_subsample(sess, cur_batch, FLAGS)
        elif FLAGS.mode == 'ce_train':
            loss_value = train_graph.run_ce_training(sess, cur_batch, FLAGS)
        total_loss += loss_value

        if step % 100==0:
            print('{} '.format(step), end="")
            sys.stdout.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % ftDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
            print()
            duration = time.time() - start_time
            print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
            sys.stdout.flush()
            log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
            log_file.flush()
            total_loss = 0.0

            best_accu, best_bleu = validate_and_save(sess, saver, FLAGS, log_file,
                    devDataStream, valid_graph, path_prefix, best_accu, best_bleu)
    print('=====End the fine tuning.')
    sys.stdout.flush()
    return best_accu, best_bleu


def enrich_options(options):
    if not options.__dict__.has_key("finetune_path"):
        options.__dict__["finetune_path"] = ""

    if not options.__dict__.has_key("CE_loss"):
        options.__dict__["CE_loss"] = False

    if not options.__dict__.has_key("infile_format"):
        options.__dict__["infile_format"] = "plain"

    if not options.__dict__.has_key("with_target_lattice"):
        options.__dict__["with_target_lattice"] = False

    if not options.__dict__.has_key("add_first_word_prob_for_phrase"):
        options.__dict__["add_first_word_prob_for_phrase"] = False

    if not options.__dict__.has_key("pretrain_with_max_matching"):
        options.__dict__["pretrain_with_max_matching"] = False

    if not options.__dict__.has_key("reward_type"):
        options.__dict__["reward_type"] = "bleu"

    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="3"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    FLAGS = enrich_options(FLAGS)

    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
