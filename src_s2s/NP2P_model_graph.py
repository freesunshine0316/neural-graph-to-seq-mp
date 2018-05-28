import tensorflow as tf
import encoder_utils
import matching_encoder_utils
import phrase_projection_layer_utils
import generator_utils
from tensorflow.python.ops import variable_scope
import numpy as np
import padding_utils
import random

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
cc = SmoothingFunction()

class ModelGraph(object):
    def __init__(self, enc_word_vocab=None, dec_word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, options=None, mode='ce_train'):

        # here 'mode', whose value can be:
        #  'ce_train',
        #  'rl_train',
        #  'evaluate',
        #  'evaluate_bleu',
        #  'decode'.
        # it is different from 'mode_gen' in generator_utils.py
        # value of 'mode_gen' can be 'ce_train', 'loss', 'greedy' or 'sample'
        self.mode = mode

        # is_training controls whether to use dropout
        is_training = True if mode in ('ce_train', ) else False

        self.options = options
        self.enc_word_vocab = enc_word_vocab
        self.dec_word_vocab = dec_word_vocab

        # create placeholders
        self.create_placeholders(options)

        # create encoder
        if options.two_sent_inputs: # take two sentences as inputs
            self.encoder = matching_encoder_utils.MatchingEncoder(self, options,
                                word_vocab=enc_word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab)
        else: # take one sentence as input
            self.encoder = encoder_utils.SeqEncoder(self, options,
                                word_vocab=enc_word_vocab, char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab)

        # encode the input instance
        self.encode_dim, self.encode_hiddens, self.init_decoder_state = self.encoder.encode(is_training=is_training)

        # project to phrase representation
        if options.with_phrase_projection:
            phrase_projection_layer = phrase_projection_layer_utils.PhraseProjectionLayer(self)
            self.phrase_representations = phrase_projection_layer.project_to_phrase_representation(self.encode_hiddens)
            self.encode_dim = 2 * self.encode_dim
        else:
            self.phrase_representations = self.encode_hiddens
            self.phrase_idx = self.in_passage_words
            self.phrase_lengths = self.passage_lengths

        phrase_length_max = tf.shape(self.phrase_idx)[1]
        self.phrase_mask = tf.sequence_mask(self.phrase_lengths, phrase_length_max, dtype=tf.float32)

        loss_weights = tf.sequence_mask(self.answer_lengths, options.max_answer_len, dtype=tf.float32) # [batch_size, gen_steps]

        with variable_scope.variable_scope("generator"):
            # create generator
            self.generator = generator_utils.CovCopyAttenGen(self, options, dec_word_vocab)
            # calculate encoder_features
            self.encoder_features = self.generator.calculate_encoder_features(self.phrase_representations, self.encode_dim)

            if mode == 'decode':
                self.context_t_1 = tf.placeholder(tf.float32, [None, self.encode_dim], name='context_t_1') # [batch_size, encode_dim]
                self.coverage_t_1 = tf.placeholder(tf.float32, [None, None], name='coverage_t_1') # [batch_size, encode_dim]
                self.word_t = tf.placeholder(tf.int32, [None], name='word_t') # [batch_size]

                (self.state_t, self.context_t, self.coverage_t, self.attn_dist_t, self.p_gen_t, self.ouput_t,
                    self.topk_log_probs, self.topk_ids, self.greedy_prediction, self.multinomial_prediction) = self.generator.decode_mode(
                        dec_word_vocab, options.beam_size, self.init_decoder_state, self.context_t_1, self.coverage_t_1, self.word_t,
                        self.phrase_representations, self.encoder_features, self.phrase_idx, self.phrase_mask)
                # not buiding training op for this mode
                return
            elif mode == 'evaluate_bleu':
                _, _, self.greedy_words = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations, self.encoder_features,
                    self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                    self.gen_input_words, self.in_answer_words, loss_weights, mode_gen='greedy')
                # not buiding training op for this mode
                return
            elif mode in ('ce_train', 'evaluate', ):
                self.accu, self.loss, _ = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations, self.encoder_features,
                    self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                    self.gen_input_words, self.in_answer_words, loss_weights, mode_gen='ce_train')
                if mode == 'evaluate': return # not buiding training op for evaluation
            elif mode == 'rl_train':
                _, self.loss, _ = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations,self.encoder_features,
                        self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                        self.gen_input_words, self.in_answer_words, loss_weights, mode_gen='loss')

                tf.get_variable_scope().reuse_variables()

                _, _, self.sampled_words = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations,self.encoder_features,
                        self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                        self.gen_input_words, self.in_answer_words, None, mode_gen='sample')

                _, _, self.greedy_words = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations,self.encoder_features,
                        self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                        self.gen_input_words, self.in_answer_words, None, mode_gen='greedy')
            elif mode == 'rl_train_for_phrase':
                _, self.loss, _ = self.generator.train_mode(dec_word_vocab, self.encode_dim, self.phrase_representations,self.encoder_features,
                        self.phrase_idx, self.phrase_mask, self.init_decoder_state,
                        self.gen_input_words, self.in_answer_words, loss_weights, mode_gen='loss')



        if options.optimize_type == 'adadelta':
            clipper = 50
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        elif options.optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)

    def create_placeholders(self, options):
        # build placeholder for input passage/article
        self.passage_lengths = tf.placeholder(tf.int32, [None], name='passage_lengths')
        if options.with_word: self.in_passage_words = tf.placeholder(tf.int32, [None, None], name="in_passage_words") # [batch_size, passage_len]
        if options.with_POS: self.in_passage_POSs = tf.placeholder(tf.int32, [None, None], name="in_passage_POSs") # [batch_size, passage_len]
        if options.with_NER: self.in_passage_NERs = tf.placeholder(tf.int32, [None, None], name="in_passage_NERs") # [batch_size, passage_len]
        if options.with_char:
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,None], name="passage_char_lengths") # [batch_size, passage_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None], name="in_passage_chars") # [batch_size, passage_len, p_char_len]

        # build placeholder for question if the model tasks two input sentences
        if options.two_sent_inputs:
            self.question_lengths = tf.placeholder(tf.int32, [None], name="question_lengths")
            if options.with_word: self.in_question_words = tf.placeholder(tf.int32, [None, None], name="in_question_words") # [batch_size, question_len]
            if options.with_POS: self.in_question_POSs = tf.placeholder(tf.int32, [None, None], name="in_question_POSs") # [batch_size, question_len]
            if options.with_NER: self.in_question_NERs = tf.placeholder(tf.int32, [None, None], name="in_question_NERs") # [batch_size, question_len]
            if options.with_char:
                self.question_char_lengths = tf.placeholder(tf.int32, [None,None], name="question_char_lengths") # [batch_size, question_len]
                self.in_question_chars = tf.placeholder(tf.int32, [None, None, None], name="in_question_chars") # [batch_size, question_len, q_char_len]

        # build placeholder for phrase projection layer
        if options.with_phrase_projection:
            self.max_phrase_size = tf.placeholder(tf.int32, [], name="max_phrase_size")# a scaler, max number of phrases within a batch
            self.phrase_starts = tf.placeholder(tf.int32, [None, None], name="phrase_starts")# [batch_size, phrase_length]
            self.phrase_ends = tf.placeholder(tf.int32, [None, None], name="phrase_ends")# [batch_size, phrase_length]
            self.phrase_idx = tf.placeholder(tf.int32, [None, None], name="phrase_idx")# [batch_size, phrase_length]
            self.phrase_lengths = tf.placeholder(tf.int32, [None], name="phrase_lengths")# [batch_size]

        # build placeholder for answer
        self.gen_input_words = tf.placeholder(tf.int32, [None, options.max_answer_len], name="gen_input_words") # [batch_size, gen_steps]
        self.in_answer_words = tf.placeholder(tf.int32, [None, options.max_answer_len], name="in_answer_words") # [batch_size, gen_steps]
        self.answer_lengths = tf.placeholder(tf.int32, [None], name="answer_lengths") # [batch_size]

        # build placeholder for reinforcement learning
        self.reward = tf.placeholder(tf.float32, [None], name="reward")


    def run_greedy(self, sess, batch, options):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict
        feed_dict[self.gen_input_words] = batch.gen_input_words
        return sess.run(self.greedy_words, feed_dict)


    def run_ce_training(self, sess, batch, options, only_eval=False):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True) # reuse this function to construct feed_dict
        feed_dict[self.gen_input_words] = batch.gen_input_words
        feed_dict[self.in_answer_words] = batch.in_answer_words
        feed_dict[self.answer_lengths] = batch.answer_lengths

        if only_eval:
            return sess.run([self.accu, self.loss], feed_dict)
        else:
            return sess.run([self.train_op, self.loss], feed_dict)[1]


    def run_rl_training_2(self, sess, batch, options):
        flipp = options.flipp if options.__dict__.has_key('flipp') else 0.1

        # make feed_dict
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.gen_input_words] = batch.gen_input_words

        # get greedy and gold outputs
        greedy_output = sess.run(self.greedy_words, feed_dict)
        greedy_output = greedy_output.tolist()
        gold_output = batch.in_answer_words.tolist()

        # generate sample_output by flipping coins
        sample_output = np.copy(batch.in_answer_words)
        for i in range(batch.in_answer_words.shape[0]):
            seq_len = min(options.max_answer_len, batch.answer_lengths[i]-1) # don't change stop token '</s>'
            for j in range(seq_len):
                if greedy_output[i][j] != 0 and random.random() < flipp:
                    sample_output[i,j] = greedy_output[i][j]
        sample_output = sample_output.tolist()

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        reward = []
        for i, (sout,gout) in enumerate(zip(sample_output,greedy_output)):
            sout, slex = self.dec_word_vocab.getLexical(sout)
            gout, glex = self.dec_word_vocab.getLexical(gout)
            rl_inputs.append([int(batch.gen_input_words[i,0])]+sout[:-1])
            rl_outputs.append(sout)
            rl_input_lengths.append(len(sout))
            _, ref_lex = self.dec_word_vocab.getLexical(gold_output[i])
            slst = slex.split()
            glst = glex.split()
            rlst = ref_lex.split()
            if options.reward_type == 'bleu':
                r = sentence_bleu([rlst], slst, smoothing_function=cc.method3)
                b = sentence_bleu([rlst], glst, smoothing_function=cc.method3)
            elif options.reward_type == 'rouge':
                r = sentence_rouge(ref_lex, slex, smoothing_function=cc.method3)
                b = sentence_rouge(ref_lex, glex, smoothing_function=cc.method3)
            reward.append(r-b)
            #print('Ref: {}'.format(ref_lex.encode('utf-8','ignore')))
            #print('Sample: {}'.format(slex.encode('utf-8','ignore')))
            #print('Greedy: {}'.format(glex.encode('utf-8','ignore')))
            #print('R-B: {}'.format(reward[-1]))
            #print('-----')

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.options.max_answer_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.options.max_answer_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.reward] = reward
        feed_dict[self.gen_input_words] = rl_inputs
        feed_dict[self.in_answer_words] = rl_outputs
        feed_dict[self.answer_lengths] = rl_input_lengths

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


    def run_rl_training(self, sess, batch, options):
        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.gen_input_words] = batch.gen_input_words

        sample_output, greedy_output = sess.run(
                [self.sampled_words, self.greedy_words], feed_dict)

        sample_output = sample_output.tolist()
        greedy_output = greedy_output.tolist()

        rl_inputs = []
        rl_outputs = []
        rl_input_lengths = []
        reward = []
        for i, (sout,gout) in enumerate(zip(sample_output,greedy_output)):
            sout, slex = self.dec_word_vocab.getLexical(sout)
            gout, glex = self.dec_word_vocab.getLexical(gout)
            rl_inputs.append([int(batch.gen_input_words[i,0])]+sout[:-1])
            rl_outputs.append(sout)
            rl_input_lengths.append(len(sout))
            ref_lex = batch.instances[i][1].tokText
            #r = metric_utils.evaluate_captions([ref_lex,],[slex,])
            #b = metric_utils.evaluate_captions([ref_lex,],[glex,])
            slst = slex.split()
            glst = glex.split()
            rlst = ref_lex.split()
            if options.reward_type == 'bleu':
                r = sentence_bleu([rlst], slst, smoothing_function=cc.method3)
                b = sentence_bleu([rlst], glst, smoothing_function=cc.method3)
            elif options.reward_type == 'rouge':
                r = sentence_rouge(ref_lex, slex, smoothing_function=cc.method3)
                b = sentence_rouge(ref_lex, glex, smoothing_function=cc.method3)
            reward.append(r-b)
            #print('Ref: {}'.format(ref_lex.encode('utf-8','ignore')))
            #print('Sample: {}'.format(slex.encode('utf-8','ignore')))
            #print('Greedy: {}'.format(glex.encode('utf-8','ignore')))
            #print('R-B: {}'.format(reward[-1]))
            #print('-----')

        rl_inputs = padding_utils.pad_2d_vals(rl_inputs, len(rl_inputs), self.options.max_answer_len)
        rl_outputs = padding_utils.pad_2d_vals(rl_outputs, len(rl_outputs), self.options.max_answer_len)
        rl_input_lengths = np.array(rl_input_lengths, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        assert rl_inputs.shape == rl_outputs.shape

        feed_dict = self.run_encoder(sess, batch, options, only_feed_dict=True)
        feed_dict[self.reward] = reward
        feed_dict[self.gen_input_words] = rl_inputs
        feed_dict[self.in_answer_words] = rl_outputs
        feed_dict[self.answer_lengths] = rl_input_lengths

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def run_encoder(self, sess, batch, options, only_feed_dict=False):
        feed_dict = {}
        feed_dict[self.passage_lengths] = batch.sent1_length
        if options.with_word: feed_dict[self.in_passage_words] = batch.sent1_word
        if options.with_char:
            feed_dict[self.passage_char_lengths] = batch.sent1_char_lengths
            feed_dict[self.in_passage_chars] = batch.sent1_char
        if options.with_POS: feed_dict[self.in_passage_POSs] = batch.sent1_POS
        if options.with_NER: self.in_passage_NERs = batch.sent1_NER

        if only_feed_dict:
            return feed_dict

        return sess.run([self.phrase_representations, self.init_decoder_state, self.encoder_features, self.phrase_idx,
                          self.phrase_mask], feed_dict)

if __name__ == '__main__':
    summary = " Tokyo is the one of the biggest city in the world."
    reference = "The capital of Japan, Tokyo, is the center of Japanese economy."

