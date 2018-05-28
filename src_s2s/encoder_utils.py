import tensorflow as tf
import match_utils

def collect_final_step_lstm(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')

class SeqEncoder(object):
    def __init__(self, placeholders, options, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None):

        self.options = options

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.POS_vocab = POS_vocab
        self.NER_vocab = NER_vocab

        self.passage_lengths = placeholders.passage_lengths #tf.placeholder(tf.int32, [None])
        if options.with_word:
            self.in_passage_words = placeholders.in_passage_words #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        if options.with_char:
            self.passage_char_lengths = placeholders.passage_char_lengths
            #tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_passage_chars = placeholders.in_passage_chars
            #tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]

        if options.with_POS:
            self.in_passage_POSs = placeholders.in_passage_POSs #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        if options.with_NER:
            self.in_passage_NERs = placeholders.in_passage_NERs #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

    def encode(self, is_training=True):
        options = self.options

        # ======word representation layer======
        in_passage_repres = []
        input_dim = 0
        if options.with_word and self.word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.variable_scope("embedding"), tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                                  initializer=tf.constant(self.word_vocab.word_vecs), dtype=tf.float32)

            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words)
            # [batch_size, passage_len, word_dim]
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_passage_words)
            batch_size = input_shape[0]
            passage_len = input_shape[1]
            input_dim += self.word_vocab.word_dim

        if options.with_char and self.char_vocab is not None:
            input_shape = tf.shape(self.in_passage_chars)
            batch_size = input_shape[0]
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = self.char_vocab.word_dim
            self.char_embedding = tf.get_variable("char_embedding",
                    initializer=tf.constant(self.char_vocab.word_vecs), dtype=tf.float32)
            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                    self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(options.char_lstm_dim)
                # dropout
                if is_training: char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell,
                        output_keep_prob=(1 - options.dropout_rate))
                char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])
                # passage representation
                passage_char_outputs = tf.nn.dynamic_rnn(char_lstm_cell, in_passage_char_repres,
                        sequence_length=passage_char_lengths,dtype=tf.float32)[0]
                # [batch_size*question_len, q_char_len, char_lstm_dim]
                passage_char_outputs = collect_final_step_lstm(passage_char_outputs, passage_char_lengths-1)
                passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, options.char_lstm_dim])

            in_passage_repres.append(passage_char_outputs)
            input_dim += options.char_lstm_dim

        in_passage_repres = tf.concat(in_passage_repres, 2) # [batch_size, passage_len, dim]

        if options.compress_input: # compress input word vector into smaller vectors
            w_compress = tf.get_variable("w_compress_input", [input_dim, options.compress_input_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress_input", [options.compress_input_dim], dtype=tf.float32)

            in_passage_repres = tf.reshape(in_passage_repres, [-1, input_dim])
            in_passage_repres = tf.matmul(in_passage_repres, w_compress) + b_compress
            in_passage_repres = tf.tanh(in_passage_repres)
            in_passage_repres = tf.reshape(in_passage_repres, [batch_size, passage_len, options.compress_input_dim])
            input_dim = options.compress_input_dim

        if is_training:
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))
        else:
            in_passage_repres = tf.multiply(in_passage_repres, (1 - options.dropout_rate))

        passage_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]

        # sequential context matching
        passage_forward = None
        passage_backward = None
        all_passage_representation = []
        passage_dim = 0
        with_lstm = True
        if with_lstm:
            with tf.variable_scope('biLSTM'):
                cur_in_passage_repres = in_passage_repres
                for i in xrange(options.context_layer_num):
                    with tf.variable_scope('layer-{}'.format(i)):
                        with tf.variable_scope('context_represent'):
                            # parameters
                            context_lstm_cell_fw = tf.contrib.rnn.LSTMCell(options.context_lstm_dim)
                            context_lstm_cell_bw = tf.contrib.rnn.LSTMCell(options.context_lstm_dim)
                            if is_training:
                                context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - options.dropout_rate))
                                context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - options.dropout_rate))

                            # passage representation
                            ((passage_context_representation_fw, passage_context_representation_bw),
                                (passage_forward, passage_backward)) = tf.nn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, cur_in_passage_repres, dtype=tf.float32,
                                        sequence_length=self.passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                            if options.direction == 'forward':
                                # [batch_size, passage_len, context_lstm_dim]
                                cur_in_passage_repres = passage_context_representation_fw
                                passage_dim += options.context_lstm_dim
                            elif options.direction == 'backward':
                                # [batch_size, passage_len, context_lstm_dim]
                                cur_in_passage_repres = passage_context_representation_bw
                                passage_dim += options.context_lstm_dim
                            elif options.direction == 'bidir':
                                # [batch_size, passage_len, 2*context_lstm_dim]
                                cur_in_passage_repres = tf.concat(
                                        [passage_context_representation_fw, passage_context_representation_bw], 2)
                                passage_dim += 2 * options.context_lstm_dim
                            else:
                                assert False
                            all_passage_representation.append(cur_in_passage_repres)


        all_passage_representation = tf.concat(all_passage_representation, 2) # [batch_size, passage_len, passage_dim]

        if is_training:
            all_passage_representation = tf.nn.dropout(all_passage_representation, (1 - options.dropout_rate))
        else:
            all_passage_representation = tf.multiply(all_passage_representation, (1 - options.dropout_rate))

        # ======Highway layer======
        if options.with_match_highway:
            with tf.variable_scope("context_highway"):
                all_passage_representation = match_utils.multi_highway_layer(all_passage_representation,
                                                                                    passage_dim,options.highway_layer_num)

        all_passage_representation = all_passage_representation * tf.expand_dims(passage_mask, axis=-1)

        # initial state for the LSTM decoder
        #'''
        with tf.variable_scope('initial_state_for_decoder'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [2*options.context_lstm_dim, options.gen_hidden_size], dtype=tf.float32)
            w_reduce_h = tf.get_variable('w_reduce_h', [2*options.context_lstm_dim, options.gen_hidden_size], dtype=tf.float32)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [options.gen_hidden_size], dtype=tf.float32)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [options.gen_hidden_size], dtype=tf.float32)

            old_c = tf.concat(values=[passage_forward.c, passage_backward.c], axis=1)
            old_h = tf.concat(values=[passage_forward.h, passage_backward.h], axis=1)
            new_c = tf.nn.tanh(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.tanh(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)

            init_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        '''
        new_c = tf.zeros([batch_size, options.gen_hidden_size])
        new_h = tf.zeros([batch_size, options.gen_hidden_size])
        init_state = LSTMStateTuple(new_c, new_h)
        '''
        return (passage_dim, all_passage_representation, init_state)

