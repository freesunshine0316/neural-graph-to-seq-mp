import tensorflow as tf
import match_utils
from tensorflow.python.ops import rnn

class MatchingEncoder(object):
    def __init__(self, placeholders, options, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None):

        self.options = options

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.POS_vocab = POS_vocab
        self.NER_vocab = NER_vocab

        self.question_lengths = placeholders.question_lengths #tf.placeholder(tf.int32, [None])
        self.passage_lengths = placeholders.passage_lengths #tf.placeholder(tf.int32, [None])
        if options.with_word:
            self.in_question_words = placeholders.in_question_words #tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_words = placeholders.in_passage_words #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        if options.with_char:
            self.question_char_lengths = placeholders.question_char_lengths #tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
            self.passage_char_lengths = placeholders.passage_char_lengths #tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_question_chars = placeholders.in_question_chars #tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
            self.in_passage_chars = placeholders.in_passage_chars #tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]

        if options.with_POS:
            self.in_question_POSs = placeholders.in_question_POSs #tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_POSs = placeholders.in_passage_POSs #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

        if options.with_NER:
            self.in_question_NERs = placeholders.in_question_NERs #tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
            self.in_passage_NERs = placeholders.in_passage_NERs #tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]

    def encode(self, is_training=True):
        options = self.options

        # ======word representation layer======
        in_question_repres = []
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

            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += self.word_vocab.word_dim

        if options.with_char and self.char_vocab is not None:
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = self.char_vocab.word_dim
            self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(self.char_vocab.word_vecs), dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(options.char_lstm_dim)
                # dropout
                if is_training: char_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - options.dropout_rate))
                char_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([char_lstm_cell])

                # question_representation
                question_char_outputs = tf.nn.dynamic_rnn(char_lstm_cell, in_question_char_repres,
                        sequence_length=question_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                question_char_outputs = question_char_outputs[:,-1,:]
                question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, options.char_lstm_dim])

                tf.get_variable_scope().reuse_variables()
                # passage representation
                passage_char_outputs = tf.nn.dynamic_rnn(char_lstm_cell, in_passage_char_repres,
                        sequence_length=passage_char_lengths,dtype=tf.float32)[0] # [batch_size*question_len, q_char_len, char_lstm_dim]
                passage_char_outputs = passage_char_outputs[:,-1,:]
                passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, options.char_lstm_dim])

            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)
            input_dim += options.char_lstm_dim

        if options.with_POS and self.POS_vocab is not None:
            self.POS_embedding = tf.get_variable("POS_embedding", initializer=tf.constant(self.POS_vocab.word_vecs), dtype=tf.float32)

            in_question_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_question_POSs) # [batch_size, question_len, POS_dim]
            in_passage_POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.in_passage_POSs) # [batch_size, passage_len, POS_dim]
            in_question_repres.append(in_question_POS_repres)
            in_passage_repres.append(in_passage_POS_repres)

            input_shape = tf.shape(self.in_question_POSs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_POSs)
            passage_len = input_shape[1]
            input_dim += self.POS_vocab.word_dim

        if options.with_NER and self.NER_vocab is not None:
            self.NER_embedding = tf.get_variable("NER_embedding", initializer=tf.constant(self.NER_vocab.word_vecs), dtype=tf.float32)

            in_question_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_question_NERs) # [batch_size, question_len, NER_dim]
            in_passage_NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.in_passage_NERs) # [batch_size, passage_len, NER_dim]
            in_question_repres.append(in_question_NER_repres)
            in_passage_repres.append(in_passage_NER_repres)

            input_shape = tf.shape(self.in_question_NERs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_NERs)
            passage_len = input_shape[1]
            input_dim += self.NER_vocab.word_dim

        in_question_repres = tf.concat(in_question_repres, 2) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(in_passage_repres, 2) # [batch_size, passage_len, dim]

        if options.compress_input: # compress input word vector into smaller vectors
            w_compress = tf.get_variable("w_compress_input", [input_dim, options.compress_input_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress_input", [options.compress_input_dim], dtype=tf.float32)

            in_question_repres = tf.reshape(in_question_repres, [-1, input_dim])
            in_question_repres = tf.matmul(in_question_repres, w_compress) + b_compress
            in_question_repres = tf.tanh(in_question_repres)
            in_question_repres = tf.reshape(in_question_repres, [batch_size, question_len, options.compress_input_dim])

            in_passage_repres = tf.reshape(in_passage_repres, [-1, input_dim])
            in_passage_repres = tf.matmul(in_passage_repres, w_compress) + b_compress
            in_passage_repres = tf.tanh(in_passage_repres)
            in_passage_repres = tf.reshape(in_passage_repres, [batch_size, passage_len, options.compress_input_dim])
            input_dim = options.compress_input_dim

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))
        else:
            in_question_repres = tf.multiply(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.multiply(in_passage_repres, (1 - options.dropout_rate))

        passage_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)

        # ======Filter layer======
        cosine_matrix = match_utils.cal_relevancy_matrix(in_question_repres, in_passage_repres)
        cosine_matrix = match_utils.mask_relevancy_matrix(cosine_matrix, question_mask, passage_mask)
#         relevancy_matrix = tf.select(tf.greater(cosine_matrix,
#                                     tf.scalar_mul(filter_layer_threshold, tf.ones_like(cosine_matrix, dtype=tf.float32))),
#                                     cosine_matrix, tf.zeros_like(cosine_matrix, dtype=tf.float32)) # [batch_size, passage_len, question_len]
        raw_in_passage_repres = in_passage_repres
        if options.with_filter_layer:
            relevancy_matrix = cosine_matrix # [batch_size, passage_len, question_len]
            relevancy_degrees = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size, passage_len]
            relevancy_degrees = tf.expand_dims(relevancy_degrees,axis=-1) # [batch_size, passage_len, 'x']
            in_passage_repres = tf.multiply(in_passage_repres, relevancy_degrees)

        # =======Context Representation Layer & Multi-Perspective matching layer=====
        all_question_aware_representatins = []
        question_aware_dim = 0
        if options.with_word_match:
            with tf.variable_scope('word_level_matching'):
                (word_match_vectors, word_match_dim) = match_utils.match_passage_with_question(raw_in_passage_repres,
                                None, passage_mask, in_question_repres, None,question_mask, input_dim, with_full_matching=False,
                                with_attentive_matching=options.with_attentive_matching,
                                with_max_attentive_matching=options.with_max_attentive_matching,
                                with_maxpooling_matching=options.with_maxpooling_matching,
                                with_local_attentive_matching=options.with_local_attentive_matching, win_size=options.win_size,
                                with_forward_match=True, with_backward_match=False, match_options=options)
                all_question_aware_representatins.extend(word_match_vectors)
                question_aware_dim += word_match_dim
        # lex decomposition
        if options.with_lex_decomposition:
            lex_decomposition = match_utils.cal_linear_decomposition_representation(raw_in_passage_repres, self.passage_lengths,
                                        cosine_matrix,is_training, options.lex_decompsition_dim, options.dropout_rate)
            all_question_aware_representatins.append(lex_decomposition)
            if options.lex_decompsition_dim== -1: question_aware_dim += 2 * input_dim
            else: question_aware_dim += 2* options.lex_decompsition_dim

        if options.with_question_passage_word_feature:
            all_question_aware_representatins.append(raw_in_passage_repres)

            att_question_representation = match_utils.calculate_cosine_weighted_question_representation(
                                             in_question_repres, cosine_matrix)
            all_question_aware_representatins.append(att_question_representation)
            question_aware_dim += 2*input_dim

        # sequential context matching
        question_forward = None
        question_backward = None
        passage_forward = None
        passage_backward = None
        if options.with_sequential_match:
            with tf.variable_scope('context_MP_matching'):
                cur_in_question_repres = in_question_repres
                cur_in_passage_repres = in_passage_repres
                for i in xrange(options.context_layer_num):
                    with tf.variable_scope('layer-{}'.format(i)):
                        with tf.variable_scope('context_represent'):
                            # parameters
                            context_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(options.context_lstm_dim)
                            context_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(options.context_lstm_dim)
                            if is_training:
                                context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - options.dropout_rate))
                                context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - options.dropout_rate))
#                             context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
#                             context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                            # question representation
                            ((question_context_representation_fw, question_context_representation_bw),
                                (question_forward, question_backward)) = tf.nn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, cur_in_question_repres, dtype=tf.float32,
                                        sequence_length=self.question_lengths) # [batch_size, question_len, context_lstm_dim]
                            cur_in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

                            # passage representation
                            tf.get_variable_scope().reuse_variables()
                            ((passage_context_representation_fw, passage_context_representation_bw),
                                (passage_forward, passage_backward)) = tf.nn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, cur_in_passage_repres, dtype=tf.float32,
                                        sequence_length=self.passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                            cur_in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)

                        # Multi-perspective matching
                        with tf.variable_scope('MP_matching'):
                            (matching_vectors, matching_dim) = match_utils.match_passage_with_question(
                                passage_context_representation_fw, passage_context_representation_bw, passage_mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                options.context_lstm_dim,
                                with_full_matching=options.with_full_matching,
                                with_attentive_matching=options.with_attentive_matching,
                                with_max_attentive_matching=options.with_max_attentive_matching,
                                with_maxpooling_matching=options.with_maxpooling_matching,
                                with_local_attentive_matching=options.with_local_attentive_matching, win_size=options.win_size,
                                with_forward_match=options.with_forward_match,
                                with_backward_match=options.with_backward_match, match_options=options)
                            all_question_aware_representatins.extend(matching_vectors)
                            question_aware_dim += matching_dim

        all_question_aware_representatins = tf.concat(all_question_aware_representatins, 2) # [batch_size, passage_len, dim]

        if is_training:
            all_question_aware_representatins = tf.nn.dropout(all_question_aware_representatins, (1 - options.dropout_rate))
        else:
            all_question_aware_representatins = tf.multiply(all_question_aware_representatins, (1 - options.dropout_rate))

        # ======Highway layer======
        if options.with_match_highway:
            with tf.variable_scope("matching_highway"):
                all_question_aware_representatins = match_utils.multi_highway_layer(all_question_aware_representatins,
                                                                                    question_aware_dim,options.highway_layer_num)

        #========Aggregation Layer======
        if not options.with_aggregation:
            aggregation_representation = all_question_aware_representatins
            aggregation_dim = question_aware_dim
        else:
            aggregation_representation = []
            aggregation_dim = 0
            aggregation_input = all_question_aware_representatins
            with tf.variable_scope('aggregation_layer'):
                for i in xrange(options.aggregation_layer_num):
                    with tf.variable_scope('layer-{}'.format(i)):
                        aggregation_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(options.aggregation_lstm_dim)
                        aggregation_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(options.aggregation_lstm_dim)
                        if is_training:
                            aggregation_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_fw,
                                                                                     output_keep_prob=(1 - options.dropout_rate))
                            aggregation_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(aggregation_lstm_cell_bw,
                                                                                     output_keep_prob=(1 - options.dropout_rate))
                        aggregation_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_fw])
                        aggregation_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([aggregation_lstm_cell_bw])

                        cur_aggregation_representation, _ = rnn.bidirectional_dynamic_rnn(
                            aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input,
                            dtype=tf.float32, sequence_length=self.passage_lengths)
                        cur_aggregation_representation = tf.concat(cur_aggregation_representation, 2) # [batch_size, passage_len, 2*aggregation_lstm_dim]
                        aggregation_representation.append(cur_aggregation_representation)
                        aggregation_dim += 2* options.aggregation_lstm_dim
                        aggregation_input = cur_aggregation_representation

            aggregation_representation = tf.concat(aggregation_representation, 2)
            aggregation_representation = tf.concat([aggregation_representation, all_question_aware_representatins], 2)
            aggregation_dim += question_aware_dim

        # ======Highway layer======
        if options.with_aggregation_highway:
            with tf.variable_scope("aggregation_highway"):
                aggregation_representation = match_utils.multi_highway_layer(aggregation_representation, aggregation_dim,
                                                                             options.highway_layer_num)

        #========output Layer=========
        encode_size = aggregation_dim + input_dim
        encode_hiddens = tf.concat([aggregation_representation, in_passage_repres], 2) # [batch_size, passage_len, enc_size]
        encode_hiddens = encode_hiddens * tf.expand_dims(passage_mask, axis=-1)

        # initial state for the LSTM decoder
        #'''
        with tf.variable_scope('initial_state_for_decoder'):
            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable('w_reduce_c', [4*options.context_lstm_dim, options.gen_hidden_size], dtype=tf.float32)
            w_reduce_h = tf.get_variable('w_reduce_h', [4*options.context_lstm_dim, options.gen_hidden_size], dtype=tf.float32)
            bias_reduce_c = tf.get_variable('bias_reduce_c', [options.gen_hidden_size], dtype=tf.float32)
            bias_reduce_h = tf.get_variable('bias_reduce_h', [options.gen_hidden_size], dtype=tf.float32)

            old_c = tf.concat(values=[question_forward.c, question_backward.c, passage_forward.c, passage_backward.c], axis=1)
            old_h = tf.concat(values=[question_forward.h, question_backward.h, passage_forward.h, passage_backward.h], axis=1)
            new_c = tf.nn.tanh(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
            new_h = tf.nn.tanh(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)

            init_state = tf.nn.LSTMStateTuple(new_c, new_h)
        '''
        new_c = tf.zeros([batch_size, options.gen_hidden_size])
        new_h = tf.zeros([batch_size, options.gen_hidden_size])
        init_state = LSTMStateTuple(new_c, new_h)
        '''
        return (encode_size, encode_hiddens, init_state)

