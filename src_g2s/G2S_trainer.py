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
import G2S_data_stream
from G2S_model_graph import ModelGraph

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
            dev_total += np.sum(cur_batch.sent_len)
        elif valid_graph.mode == 'evaluate_bleu':
            gen.extend(valid_graph.run_greedy(sess, cur_batch, options).tolist())
            ref.extend(cur_batch.sent_out.tolist())
        else:
            assert False

    if valid_graph.mode == 'evaluate':
        return {'dev_loss':dev_loss, 'dev_accu':1.0*dev_right/dev_total, 'dev_right':dev_right, 'dev_total':dev_total, }
    else:
        return {'dev_bleu':document_bleu(valid_graph.word_vocab,gen,ref,suffix), }



def main(_):
    print('Configurations:')
    print(FLAGS)

    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/G2S.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    print('Loading train set.')
    trainset, trn_node, trn_in_neigh, trn_out_neigh, trn_sent = G2S_data_stream.read_amr_file(FLAGS.train_path)
    print('Number of training samples: {}'.format(len(trainset)))

    print('Loading dev set.')
    devset, tst_node, tst_in_neigh, tst_out_neigh, tst_sent = G2S_data_stream.read_amr_file(FLAGS.test_path)
    print('Number of dev samples: {}'.format(len(devset)))

    if FLAGS.finetune_path != "":
        print('Loading finetune set.')
        ftset, ft_node, ft_in_neigh, ft_out_neigh, ft_sent = G2S_data_stream.read_amr_file(FLAGS.finetune_path)
        print('Number of finetune samples: {}'.format(len(ftset)))
    else:
        ftset, ft_node, ft_in_neigh, ft_out_neigh, ft_sent = (None, 0, 0, 0, 0)

    max_node = max(trn_node, tst_node, ft_node)
    max_in_neigh = max(trn_in_neigh, tst_in_neigh, ft_in_neigh)
    max_out_neigh = max(trn_out_neigh, tst_out_neigh, ft_out_neigh)
    max_sent = max(trn_sent, tst_sent, ft_sent)
    print('Max node number: {}, while max allowed is {}'.format(max_node, FLAGS.max_node_num))
    print('Max parent number: {}, truncated to {}'.format(max_in_neigh, FLAGS.max_in_neigh_num))
    print('Max children number: {}, truncated to {}'.format(max_out_neigh, FLAGS.max_out_neigh_num))
    print('Max answer length: {}, truncated to {}'.format(max_sent, FLAGS.max_answer_len))

    word_vocab = None
    char_vocab = None
    edgelabel_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + ".best.model"
    if os.path.exists(best_path + ".index"):
        has_pretrained_model = True
        print('!!Existing pretrained model. Loading vocabs.')
        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        char_vocab = None
        if FLAGS.with_char:
            char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
            print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
        edgelabel_vocab = Vocab(path_prefix + ".edgelabel_vocab", fileformat='txt2')
    else:
        print('Collecting vocabs.')
        (allWords, allChars, allEdgelabels) = G2S_data_stream.collect_vocabs(trainset)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of allChars: {}'.format(len(allChars)))
        print('Number of allEdgelabels: {}'.format(len(allEdgelabels)))

        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        char_vocab = None
        if FLAGS.with_char:
            char_vocab = Vocab(voc=allChars, dim=FLAGS.char_dim, fileformat='build')
            char_vocab.dump_to_txt2(path_prefix + ".char_vocab")
        edgelabel_vocab = Vocab(voc=allEdgelabels, dim=FLAGS.edgelabel_dim, fileformat='build')
        edgelabel_vocab.dump_to_txt2(path_prefix + ".edgelabel_vocab")

    print('word vocab size {}'.format(word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build DataStream ... ')
    trainDataStream = G2S_data_stream.G2SDataStream(trainset, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=True, isLoop=True, isSort=True)

    devDataStream = G2S_data_stream.G2SDataStream(devset, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    if ftset != None:
        ftDataStream = G2S_data_stream.G2SDataStream(ftset, word_vocab, char_vocab, edgelabel_vocab, options=FLAGS,
                 isShuffle=True, isLoop=True, isSort=True)
        print('Number of instances in ftDataStream: {}'.format(ftDataStream.get_num_instance()))
        print('Number of batches in ftDataStream: {}'.format(ftDataStream.get_num_batch()))

    sys.stdout.flush()

    # initialize the best bleu and accu scores for current training session
    best_accu = FLAGS.best_accu if FLAGS.__dict__.has_key('best_accu') else 0.0
    best_bleu = FLAGS.best_bleu if FLAGS.__dict__.has_key('best_bleu') else 0.0
    if best_accu > 0.0:
        print('With initial dev accuracy {}'.format(best_accu))
    if best_bleu > 0.0:
        print('With initial dev BLEU score {}'.format(best_bleu))

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(word_vocab=word_vocab, Edgelabel_vocab=edgelabel_vocab,
                                         char_vocab=char_vocab, options=FLAGS, mode=FLAGS.mode)

        assert FLAGS.mode in ('ce_train', 'rl_train', )
        valid_mode = 'evaluate' if FLAGS.mode == 'ce_train' else 'evaluate_bleu'

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, Edgelabel_vocab=edgelabel_vocab,
                                         char_vocab=char_vocab, options=FLAGS, mode=valid_mode)

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
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
                sys.stdout.flush()
                best_bleu = evaluate(sess, valid_graph, devDataStream, options=FLAGS)['dev_bleu']
                FLAGS.best_bleu = best_bleu
                namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                print('BLEU = %.4f' % best_bleu)
                sys.stdout.flush()
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
                loss_value = train_graph.run_rl_training_subsample(sess, cur_batch, FLAGS)
            elif FLAGS.mode == 'ce_train':
                loss_value = train_graph.run_ce_training(sess, cur_batch, FLAGS)
            total_loss += loss_value

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps or \
                    (trainDataStream.get_num_batch() > 10000 and (step+1)%2000 == 0):
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
    return best_accu, best_bleu


def fine_tune(sess, saver, FLAGS, log_file,
    ftDataStream, devDataStream, train_graph, valid_graph, path_prefix, best_accu, best_bleu):
    print('=====Start the fine tuning.')
    sys.stdout.flush()
    max_steps = ftDataStream.get_num_batch() * 3
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
            best_accu, best_bleu = validate_and_save(sess, saver, FLAGS, log_file,
                    devDataStream, valid_graph, path_prefix, best_accu, best_bleu)
            total_loss = 0.0
            start_time = time.time()

    print('=====End the fine tuning.')
    sys.stdout.flush()
    return best_accu, best_bleu


def enrich_options(options):
    if not options.__dict__.has_key("finetune_path"):
        options.__dict__["finetune_path"] = ""

    if not options.__dict__.has_key("CE_loss"):
        options.__dict__["CE_loss"] = False

    if not options.__dict__.has_key("reward_type"):
        options.__dict__["reward_type"] = "bleu"

    if not options.__dict__.has_key("way_init_decoder"):
        options.__dict__["way_init_decoder"] = 'zero'

    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="2"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    FLAGS = enrich_options(FLAGS)

    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
