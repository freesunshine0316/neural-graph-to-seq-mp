import tensorflow as tf
import match_utils
from tf_utils import linear


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

def graph_neigh_attn(passage_node_hidden, passage_neigh_indices, passage_neigh_mask,
        w_weight, u_weight, b_weight, v_weight,
        batch_size, node_size, neigh_size, options):
    # [batch_size, node_size, neigh_size, neighbor_vector_dim]
    passage_neigh_prev_hidden = tf.multiply(
            collect_neighbor_node_representations(passage_node_hidden, passage_neigh_indices),
            tf.expand_dims(passage_neigh_mask, axis=-1))
    ## neigh feat
    # [-1, neighbor_vector_dim]
    passage_neigh_prev_feat_1 = tf.matmul(
            tf.reshape(passage_neigh_prev_hidden, [-1, options.neighbor_vector_dim]),
            w_weight) + b_weight
    # [batch_size, node_size, neigh_size, neighbor_vector_dim]
    passage_neigh_prev_feat_1 = tf.reshape(passage_neigh_prev_feat_1,
            [batch_size, node_size, neigh_size, options.neighbor_vector_dim])

    ## self feat
    # [-1, neighbor_vector_dim]
    passage_neigh_prev_feat_2 = tf.matmul(
            tf.reshape(passage_node_hidden, [-1, options.neighbor_vector_dim]),
            u_weight)
    # [batch_size, node_size, 1, neighbor_vector_dim]
    passage_neigh_prev_feat_2 = tf.expand_dims(
            tf.reshape(passage_neigh_prev_feat_2, [batch_size, node_size, options.neighbor_vector_dim]),
            axis=2)
    ## total feat
    # [batch_size, node_size, neigh_size, neighbor_vector_dim]
    passage_neigh_prev_feat = passage_neigh_prev_feat_1 + passage_neigh_prev_feat_2

    ## attn dist
    # [batch_size, node_size, neigh_size]
    tmp = tf.reduce_sum(passage_neigh_prev_feat * v_weight, axis=-1)
    passage_neigh_attn_dist = tf.nn.softmax(tmp)

    ## weighted hidden
    # [batch_size, node_size, neighbor_vector_dim
    passage_neigh_prev_hidden_weighted = tf.reduce_sum(
            passage_neigh_prev_hidden * tf.expand_dims(passage_neigh_attn_dist, axis=-1),
            axis=2)
    return passage_neigh_attn_dist, passage_neigh_prev_hidden_weighted

def apply_neigh_attn(passage_neigh_X, passage_neigh_attn_dist):
    return tf.reduce_sum(
            passage_neigh_X * tf.expand_dims(passage_neigh_attn_dist, axis=-1),
            axis=2)

class GraphEncoder(object):
    def __init__(self, word_vocab=None, edge_label_vocab=None, char_vocab=None, is_training=True, options=None):
        assert options != None

        self.passage_nodes_size = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.passage_nodes = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_nodes_size_max]
        if options.with_char:
            self.passage_nodes_chars_size = tf.placeholder(tf.int32, [None, None])
            self.passage_nodes_chars = tf.placeholder(tf.int32, [None, None, None])

        # [batch_size, passage_nodes_size_max, passage_neighbors_size_max]
        self.passage_in_neighbor_indices = tf.placeholder(tf.int32, [None, None, None])
        self.passage_in_neighbor_edges = tf.placeholder(tf.int32, [None, None, None])
        self.passage_in_neighbor_mask = tf.placeholder(tf.float32, [None, None, None])

        # [batch_size, passage_nodes_size_max, passage_neighbors_size_max]
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
        # [batch_size, passage_nodes_size_max]
        self.passage_nodes_mask = tf.sequence_mask(self.passage_nodes_size, passage_nodes_size_max, dtype=tf.float32)

        # embeddings
        word_vec_trainable = True
        cur_device = '/gpu:0'
        if options.fix_word_vec:
            word_vec_trainable = False
            cur_device = '/cpu:0'
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

        if options.compress_input: # compress input word vector into smaller vectors
            w_compress = tf.get_variable("w_compress_input", [input_dim, options.node_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress_input", [options.node_dim], dtype=tf.float32)

            passage_node_representation = tf.reshape(passage_node_representation, [-1, input_dim])
            passage_node_representation = tf.matmul(passage_node_representation, w_compress) + b_compress
            passage_node_representation = tf.tanh(passage_node_representation)
            passage_node_representation = tf.reshape(passage_node_representation, \
                    [batch_size, passage_nodes_size_max, options.node_dim])
        else:
            assert False

        if is_training:
            passage_node_representation = tf.nn.dropout(passage_node_representation, (1 - options.dropout_rate))
        else:
            passage_node_representation = tf.multiply(passage_node_representation, (1 - options.dropout_rate))


        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                passage_node_representation = match_utils.multi_highway_layer(passage_node_representation, options.node_dim,
                                                                              options.highway_layer_num)

        with tf.variable_scope('graph_encoder'):
            # =========== in neighbor
            passage_in_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
                    self.passage_in_neighbor_edges)
            # [batch_size, passage_len, passage_neighbors_size_max, edge_dim]
            passage_in_neighbor_node_representations = collect_neighbor_node_representations(
                                                    passage_node_representation, self.passage_in_neighbor_indices)
            # [batch_size, passage_len, passage_neighbors_size_max, node_dim]

            passage_in_neighbor_representations = tf.concat( \
                    [passage_in_neighbor_node_representations, passage_in_neighbor_edge_representations], 3)
            passage_in_neighbor_representations = tf.multiply(passage_in_neighbor_representations,
                    tf.expand_dims(self.passage_in_neighbor_mask, axis=-1))
            # [batch_size, passage_len, passage_neighbors_size_max, node_dim + edge_dim]

            # ============ out neighbor
            passage_out_neighbor_edge_representations = tf.nn.embedding_lookup(self.edge_embedding,
                    self.passage_out_neighbor_edges)
            # [batch_size, passage_len, passage_neighbors_size_max, edge_dim]
            passage_out_neighbor_node_representations = collect_neighbor_node_representations(
                                                    passage_node_representation, self.passage_out_neighbor_indices)
            # [batch_size, passage_len, passage_neighbors_size_max, node_dim]

            passage_out_neighbor_representations = tf.concat( \
                    [passage_out_neighbor_node_representations, passage_out_neighbor_edge_representations], 3)
            passage_out_neighbor_representations = tf.multiply(passage_out_neighbor_representations,
                    tf.expand_dims(self.passage_out_neighbor_mask, axis=-1))
            # [batch_size, passage_len, passage_neighbors_size_max, node_dim + edge_dim]

            # =====compress neighbor_representations
            compress_vector_dim = options.neighbor_vector_dim
            w_compress = tf.get_variable("w_compress", [options.node_dim + edge_dim, compress_vector_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress", [compress_vector_dim], dtype=tf.float32)

            passage_in_neighbor_representations = tf.reshape(passage_in_neighbor_representations,
                    [-1, options.node_dim + edge_dim])
            passage_in_neighbor_representations = tf.matmul(passage_in_neighbor_representations, w_compress) + b_compress
            passage_in_neighbor_representations = tf.tanh(passage_in_neighbor_representations)
            passage_in_neighbor_representations = tf.reshape(passage_in_neighbor_representations,
                    [batch_size, passage_nodes_size_max, passage_in_neighbors_size_max, options.neighbor_vector_dim])


            passage_out_neighbor_representations = tf.reshape(passage_out_neighbor_representations,
                    [-1, options.node_dim + edge_dim])
            passage_out_neighbor_representations = tf.matmul(passage_out_neighbor_representations, w_compress) + b_compress
            passage_out_neighbor_representations = tf.tanh(passage_out_neighbor_representations)
            passage_out_neighbor_representations = tf.reshape(passage_out_neighbor_representations,
                    [batch_size, passage_nodes_size_max, passage_out_neighbors_size_max, options.neighbor_vector_dim])

            # assume each node has a neighbor vector, and it is None at the beginning
            passage_node_hidden = tf.zeros([batch_size, passage_nodes_size_max, options.neighbor_vector_dim])
            passage_node_cell = tf.zeros([batch_size, passage_nodes_size_max, options.neighbor_vector_dim])

            w_in_ingate = tf.get_variable("w_in_ingate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_in_ingate = tf.get_variable("u_in_ingate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            b_ingate = tf.get_variable("b_in_ingate", [options.neighbor_vector_dim], dtype=tf.float32)

            w_out_ingate = tf.get_variable("w_out_ingate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_out_ingate = tf.get_variable("u_out_ingate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)

            w_in_forgetgate = tf.get_variable("w_in_forgetgate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_in_forgetgate = tf.get_variable("u_in_forgetgate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            b_forgetgate = tf.get_variable("b_in_forgetgate",
                    [options.neighbor_vector_dim], dtype=tf.float32)

            w_out_forgetgate = tf.get_variable("w_out_forgetgate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_out_forgetgate = tf.get_variable("u_out_forgetgate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)

            w_in_outgate = tf.get_variable("w_in_outgate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_in_outgate = tf.get_variable("u_in_outgate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            b_outgate = tf.get_variable("b_in_outgate",
                    [options.neighbor_vector_dim], dtype=tf.float32)

            w_out_outgate = tf.get_variable("w_out_outgate",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_out_outgate = tf.get_variable("u_out_outgate",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)

            w_in_cell = tf.get_variable("w_in_cell",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_in_cell = tf.get_variable("u_in_cell",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            b_cell = tf.get_variable("b_in_cell",
                    [options.neighbor_vector_dim], dtype=tf.float32)

            w_out_cell = tf.get_variable("w_out_cell",
                    [compress_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_out_cell = tf.get_variable("u_out_cell",
                    [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)

            w_in_attn = tf.get_variable("w_in_attn", [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_in_attn = tf.get_variable("u_in_attn", [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            v_in_attn = tf.get_variable("v_in_attn", [options.neighbor_vector_dim], dtype=tf.float32)
            b_in_attn = tf.get_variable("b_in_attn", [options.neighbor_vector_dim], dtype=tf.float32)

            w_out_attn = tf.get_variable("w_out_attn", [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            u_out_attn = tf.get_variable("u_out_attn", [options.neighbor_vector_dim, options.neighbor_vector_dim], dtype=tf.float32)
            v_out_attn = tf.get_variable("v_out_attn", [options.neighbor_vector_dim], dtype=tf.float32)
            b_out_attn = tf.get_variable("b_out_attn", [options.neighbor_vector_dim], dtype=tf.float32)

            # calculate question graph representation
            graph_representations = []
            for i in xrange(options.num_syntax_match_layer):
                # =============== in edge hidden
                # [batch, node, neigh], [batch, node, dim]
                passage_in_edge_attn_dist, passage_in_edge_weighted_hidden = graph_neigh_attn(
                        passage_node_hidden, self.passage_in_neighbor_indices, self.passage_in_neighbor_mask,
                        w_in_attn, u_in_attn, b_in_attn, v_in_attn,
                        batch_size, passage_nodes_size_max, passage_in_neighbors_size_max, options)
                # [-1, dim]
                passage_in_edge_weighted_hidden = tf.reshape(passage_in_edge_weighted_hidden,
                        [-1, options.neighbor_vector_dim])
                # [batch, node, dim]
                passage_in_edge_weighted_X = apply_neigh_attn(passage_in_neighbor_representations,
                        passage_in_edge_attn_dist)
                # [-1, dim]
                passage_in_edge_weighted_X = tf.reshape(passage_in_edge_weighted_X,
                        [-1, options.neighbor_vector_dim])

                # =============== out edge hidden
                # [batch, node, neigh], [batch, node, dim]
                passage_out_edge_attn_dist, passage_out_edge_weighted_hidden = graph_neigh_attn(
                        passage_node_hidden, self.passage_out_neighbor_indices, self.passage_out_neighbor_mask,
                        w_out_attn, u_out_attn, b_out_attn, v_out_attn,
                        batch_size, passage_nodes_size_max, passage_out_neighbors_size_max, options)
                # [-1, dim]
                passage_out_edge_weighted_hidden = tf.reshape(passage_out_edge_weighted_hidden,
                        [-1, options.neighbor_vector_dim])
                # [batch, node, dim]
                passage_out_edge_weighted_X = apply_neigh_attn(passage_out_neighbor_representations,
                        passage_out_edge_attn_dist)
                # [-1, dim]
                passage_out_edge_weighted_X = tf.reshape(passage_out_edge_weighted_X,
                        [-1, options.neighbor_vector_dim])

                ## ig
                passage_edge_ingate = tf.sigmoid(tf.matmul(passage_in_edge_weighted_X, w_in_ingate)
                                          + tf.matmul(passage_in_edge_weighted_hidden, u_in_ingate)
                                          + tf.matmul(passage_out_edge_weighted_X, w_out_ingate)
                                          + tf.matmul(passage_out_edge_weighted_hidden, u_out_ingate)
                                          + b_ingate)
                passage_edge_ingate = tf.reshape(passage_edge_ingate,
                        [batch_size, passage_nodes_size_max, options.neighbor_vector_dim])
                ## fg
                passage_edge_forgetgate = tf.sigmoid(tf.matmul(passage_in_edge_weighted_X, w_in_forgetgate)
                                          + tf.matmul(passage_in_edge_weighted_hidden, u_in_forgetgate)
                                          + tf.matmul(passage_out_edge_weighted_X, w_out_forgetgate)
                                          + tf.matmul(passage_out_edge_weighted_hidden, u_out_forgetgate)
                                          + b_forgetgate)
                passage_edge_forgetgate = tf.reshape(passage_edge_forgetgate,
                        [batch_size, passage_nodes_size_max, options.neighbor_vector_dim])
                ## og
                passage_edge_outgate = tf.sigmoid(tf.matmul(passage_in_edge_weighted_X, w_in_outgate)
                                          + tf.matmul(passage_in_edge_weighted_hidden, u_in_outgate)
                                          + tf.matmul(passage_out_edge_weighted_X, w_out_outgate)
                                          + tf.matmul(passage_out_edge_weighted_hidden, u_out_outgate)
                                          + b_outgate)
                passage_edge_outgate = tf.reshape(passage_edge_outgate,
                        [batch_size, passage_nodes_size_max, options.neighbor_vector_dim])
                ## input
                passage_edge_cell_input = tf.tanh(tf.matmul(passage_in_edge_weighted_X, w_in_cell)
                                          + tf.matmul(passage_in_edge_weighted_hidden, u_in_cell)
                                          + tf.matmul(passage_out_edge_weighted_X, w_out_cell)
                                          + tf.matmul(passage_out_edge_weighted_hidden, u_out_cell)
                                          + b_cell)
                passage_edge_cell_input = tf.reshape(passage_edge_cell_input,
                        [batch_size, passage_nodes_size_max, options.neighbor_vector_dim])

                passage_edge_cell = passage_edge_forgetgate * passage_node_cell + passage_edge_ingate * passage_edge_cell_input
                passage_edge_hidden = passage_edge_outgate * tf.tanh(passage_edge_cell)
                # node mask
                # [batch_size, passage_len, neighbor_vector_dim]
                passage_node_cell = tf.multiply(passage_edge_cell, tf.expand_dims(self.passage_nodes_mask, axis=-1))
                passage_node_hidden = tf.multiply(passage_edge_hidden, tf.expand_dims(self.passage_nodes_mask, axis=-1))

                graph_representations.append(passage_node_hidden)

            # decide how to use graph_representations
            self.graph_representations = graph_representations
            self.node_representations = passage_node_representation
            self.graph_hiddens = passage_node_hidden
            self.graph_cells = passage_node_cell

            self.batch_size = batch_size
