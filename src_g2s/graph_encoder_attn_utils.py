import tensorflow as tf
import match_utils
import transformer_utils as transformer

def collect_neighbor_node_representations(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    feature_dim = tf.shape(representation)[2]
    input_shape = tf.shape(positions)
    batch_size = input_shape[0]
    num_nodes = input_shape[1]
    num_neighbors = input_shape[2]
    positions_flat = tf.reshape(positions, [batch_size, num_nodes*num_neighbors])
    def singel_instance(x):
        # x[0]: [num_nodes, feature_dim]
        # x[1]: [num_nodes*num_neighbors]
        return tf.gather(x[0], x[1])
    elems = (representation, positions_flat)
    representations = tf.map_fn(singel_instance, elems, dtype=tf.float32)
    return tf.reshape(representations, [batch_size, num_nodes, num_neighbors, feature_dim])

def collect_final_step_lstm(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')

class GraphEncoder(object):
    def __init__(self, word_vocab=None, edge_label_vocab=None, char_vocab=None, is_training=True, options=None):
        assert options != None

        # placeholders
        self.passage_nodes_size = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.passage_nodes = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_nodes_size_max]
        if options.with_char:
            self.passage_nodes_chars_size = tf.placeholder(tf.int32, [None, None])
            self.passage_nodes_chars = tf.placeholder(tf.int32, [None, None, None])

        # [batch_size, passage_nodes_size_max, neighbors_size]
        self.passage_in_neighbor_indices = tf.placeholder(tf.int32, [None, None, None])
        self.passage_in_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.passage_in_neighbor_mask = tf.placeholder(tf.float32, [None, None, None])

        # [batch_size, passage_nodes_size_max, neighbors_size]
        self.passage_out_neighbor_indices = tf.placeholder(tf.int32, [None, None, None])
        self.passage_out_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.passage_out_neighbor_mask = tf.placeholder(tf.float32, [None, None, None])

        # shapes
        input_shape = tf.shape(self.passage_in_neighbor_indices)
        batch_size = input_shape[0]
        passage_nodes_size_max = input_shape[1]
        passage_in_neighbors_size_max = input_shape[2]
        passage_out_neighbors_size_max = tf.shape(self.passage_out_neighbor_indices)[2]
        if options.with_char:
            passage_nodes_chars_size_max = tf.shape(self.passage_nodes_chars)[2]

        # masks
        self.passage_nodes_mask = tf.sequence_mask(self.passage_nodes_size, passage_nodes_size_max, dtype=tf.float32)

        # embeddings
        if options.fix_word_vec:
            word_vec_trainable = False
            cur_device = '/cpu:0'
        else:
            word_vec_trainable = True
            cur_device = '/gpu:0'
        with tf.device(cur_device):
            self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                    initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

        self.edge_embedding = tf.get_variable("edge_embedding",
                initializer=tf.constant(edge_label_vocab.word_vecs), dtype=tf.float32)

        word_dim = word_vocab.word_dim
        edge_dim = edge_label_vocab.word_dim

        if options.with_char:
            self.char_embedding = tf.get_variable("char_embedding",
                    initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
            char_dim = char_vocab.word_dim

        # word representation for nodes, where each node only includes one word
        # [batch_size, passage_nodes_size_max, word_dim]
        passage_node_representation = tf.nn.embedding_lookup(self.word_embedding, self.passage_nodes)

        if options.with_char:
            # [batch_size, passage_nodes_size_max, passage_nodes_chars_size_max, char_dim]
            passage_nodes_chars_representation = tf.nn.embedding_lookup(self.char_embedding, self.passage_nodes_chars)
            passage_nodes_chars_representation = tf.reshape(passage_nodes_chars_representation,
                    shape=[batch_size*passage_nodes_size_max, passage_nodes_chars_size_max, char_dim])
            passage_nodes_chars_size = tf.reshape(self.passage_nodes_chars_size, [batch_size*passage_nodes_size_max])
            with tf.variable_scope('node_char_lstm'):
                node_char_lstm_cell = tf.contrib.rnn.LSTMCell(options.char_lstm_dim)
                node_char_lstm_cell = tf.contrib.rnn.MultiRNNCell([node_char_lstm_cell])
                # [batch_size*node_num, char_num, char_lstm_dim]
                node_char_outputs = tf.nn.dynamic_rnn(node_char_lstm_cell, passage_nodes_chars_representation,
                        sequence_length=passage_nodes_chars_size, dtype=tf.float32)[0]
                node_char_outputs = collect_final_step_lstm(node_char_outputs, passage_nodes_chars_size-1)
                # [batch_size, node_num, char_lstm_dim]
                node_char_outputs = tf.reshape(node_char_outputs, [batch_size, passage_nodes_size_max, options.char_lstm_dim])

        if options.with_char:
            input_dim = word_dim + options.char_lstm_dim
            passage_node_representation = tf.concat([passage_node_representation, node_char_outputs], 2)
        else:
            input_dim = word_dim
            passage_node_representation = passage_node_representation

        # apply the mask
        passage_node_representation = passage_node_representation * tf.expand_dims(self.passage_nodes_mask, axis=-1)

        if is_training:
            passage_node_representation = tf.nn.dropout(passage_node_representation, (1 - options.dropout_rate))
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            mode = tf.estimator.ModeKeys.TEST

        w_trans = tf.get_variable("w_trans", [input_dim, options.neighbor_vector_dim], dtype=tf.float32)
        b_trans = tf.get_variable("b_trans", [options.neighbor_vector_dim], dtype=tf.float32)

        passage_node_representation = tf.reshape(passage_node_representation, [-1, input_dim])
        passage_node_representation = tf.matmul(passage_node_representation, w_trans) + b_trans
        passage_node_representation = tf.tanh(passage_node_representation)
        passage_node_representation = tf.reshape(passage_node_representation,
                [batch_size, passage_nodes_size_max, options.neighbor_vector_dim])

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                passage_node_representation = match_utils.multi_highway_layer(passage_node_representation,
                        input_dim, options.highway_layer_num)

        with tf.variable_scope('graph_encoder'):
            # =========== neighbor relations
            # [batch_size, node_len, neighbors_size, edge_dim]
            passage_in_neighbor_rel = tf.nn.embedding_lookup(self.edge_embedding,
                    self.passage_in_neighbor_edges)
            # [batch_size, node_len, neighbors_size, edge_dim]
            passage_out_neighbor_rel = tf.nn.embedding_lookup(self.edge_embedding,
                    self.passage_out_neighbor_edges)
            # [batch_size, node_len, neighbors_size*2, edge_dim]
            passage_neighbor_rel = tf.concat([passage_in_neighbor_rel, passage_out_neighbor_rel], 2)

            # initiate hidden states with embedding representations
            passage_node_hidden = passage_node_representation

            # =========== neighbor mask
            # [batch_size, node_len, neighbors_size*2]
            passage_neighor_mask = tf.concat(
                    [self.passage_in_neighbor_mask, self.passage_out_neighbor_mask], 2)

            # calculate graph representation
            graph_representations = []
            for i in xrange(options.num_syntax_match_layer):
                with tf.variable_scope("layer_{}".format(i)):
                    # =============== in neigh hidden
                    # [batch_size, node_len, neighbors_size, neighbor_vector_dim]
                    passage_in_neighbor_hidden = collect_neighbor_node_representations(passage_node_hidden,
                            self.passage_in_neighbor_indices)
                    # =============== out edge hidden
                    # [batch_size, node_len, neighbors_size, neighbor_vector_dim]
                    passage_out_neighbor_hidden = collect_neighbor_node_representations(passage_node_hidden,
                            self.passage_out_neighbor_indices)
                    # [batch_size, node_len, neighbors_size*2, neighbor_vector_dim]
                    passage_neighbor_hidden = tf.concat(
                            [passage_in_neighbor_hidden, passage_out_neighbor_hidden], 2)

                    # [batch_size, node_len, neighbors_size*2, neighbor_vector_dim+edge_dim]
                    passage_memory = tf.concat(
                            [passage_neighbor_hidden, passage_neighbor_rel], 3)
                    passage_memory = passage_memory * tf.expand_dims(
                            passage_neighbor_mask, axis=-1)
                    # [batch_size, node_len, 1, neighbor_vector_dim]
                    passage_query = tf.expand_dims(passage_node_hidden, axis=2)

                    with tf.variable_scope("multi_head"):
                        # ========== multi head attention part
                        passage_new_query = transformer.multi_head_attention(
                                options.num_heads,
                                passage_query,
                                passage_memory,
                                mode,
                                num_units=options.neighbor_vector_dim,
                                mask=passage_neighbor_mask,
                                dropout=options.dropout_rate)
                        passage_new_query = transformer.drop_and_add(
                                passage_query,
                                passage_new_query,
                                mode,
                                dropout=options.dropout_rate)

                    with tf.variable_scope("ffn"):
                        passage_new_new_query = transformer.feed_forward(
                                transformer.norm(passage_new_query),
                                options.neighbor_vector_dim,
                                mode,
                                dropout=options.dropout_rate)
                        passage_new_new_query = transformer.drop_and_add(
                                passage_new_query,
                                passage_new_new_query,
                                mode,
                                dropout=options.dropout_rate)

                    passage_node_hidden = passage_new_new_query
                    passage_node_hidden = tf.reshape(passage_node_hidden,
                            [batch_size, passage_nodes_size_max, options.neighbor_vector_dim]

                    graph_representations.append(passage_node_hidden)

            # decide how to use graph_representations
            self.graph_representations = graph_representations
            self.node_representations = passage_node_representation
            self.graph_hiddens = passage_node_hidden
            self.graph_cells = passage_node_cell

            self.batch_size = batch_size

