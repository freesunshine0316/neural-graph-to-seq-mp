import tensorflow as tf
import match_utils

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

def collect_node_representations(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_candidate_nodes]
    def singel_instance(x):
        # x[0]: [num_nodes, feature_dim]
        # x[1]: [num_candidate_nodes]
        return tf.gather(x[0], x[1])
    elems = (representation, positions)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, num_candidate_nodes, feature_dim]


def graph_match(in_question_repres, in_passage_repres, 
                question_mask, passage_mask, edge_embedding,
                question_neighbor_indices, passage_neighbor_indices,
                question_neighbor_edges, passage_neighbor_edges,
                question_neighbor_size, passage_neighbor_size,
                neighbor_vector_dim, input_dim, edge_dim, num_syntax_match_layer,
                with_attentive_matching=True, with_max_attentive_matching=True,
                with_maxpooling_matching=True, match_options=None): 

    all_matching_vectors = []
    all_matching_dim = 0
    
    
    input_shape = tf.shape(question_neighbor_indices)
    batch_size = input_shape[0]
    question_len = input_shape[1]
    num_question_neighbors = input_shape[2]

    input_shape = tf.shape(passage_neighbor_indices)
#     batch_size = input_shape[0]
    passage_len = input_shape[1]
    num_passage_neighbors = input_shape[2]
    
    question_neighbor_mask = tf.sequence_mask(tf.reshape(question_neighbor_size, [-1]), num_question_neighbors, dtype=tf.float32)
    question_neighbor_mask = tf.reshape(question_neighbor_mask, [batch_size, question_len, num_question_neighbors])

    passage_neighbor_mask = tf.sequence_mask(tf.reshape(passage_neighbor_size, [-1]), num_passage_neighbors, dtype=tf.float32)
    passage_neighbor_mask = tf.reshape(passage_neighbor_mask, [batch_size, passage_len, num_passage_neighbors])

    question_neighbor_edge_representations = tf.nn.embedding_lookup(edge_embedding, question_neighbor_edges) 
    # [batch_size, question_len, num_question_neighbors, edge_dim]
    passage_neighbor_edge_representations = tf.nn.embedding_lookup(edge_embedding, passage_neighbor_edges) 
    # [batch_size, passage_len, num_passage_neighbors, edge_dim]
    question_neighbor_node_representations = collect_neighbor_node_representations(
                                                    in_question_repres, question_neighbor_indices)
    # [batch_size, question_len, num_question_neighbors, input_dim]
    passage_neighbor_node_representations = collect_neighbor_node_representations(
                                                    in_passage_repres, passage_neighbor_indices)
    # [batch_size, passage_len, num_passage_neighbors, input_dim]
                
    question_neighbor_representations = tf.concat(3, [question_neighbor_node_representations, question_neighbor_edge_representations])
    # [batch_size, question_len, num_question_neighbors, input_dim+ edge_dim]
    passage_neighbor_representations = tf.concat(3, [passage_neighbor_node_representations, passage_neighbor_edge_representations])
    # [batch_size, passage_len, num_passage_neighbors, input_dim + edge_dim]

    # =====compress neighbor_representations 
    compress_vector_dim = neighbor_vector_dim
    w_compress = tf.get_variable("w_compress", [input_dim + edge_dim, compress_vector_dim], dtype=tf.float32)
    b_compress = tf.get_variable("b_compress", [compress_vector_dim], dtype=tf.float32)

    question_neighbor_representations = tf.reshape(question_neighbor_representations, [-1, input_dim + edge_dim])
    question_neighbor_representations = tf.matmul(question_neighbor_representations, w_compress) + b_compress
    question_neighbor_representations = tf.tanh(question_neighbor_representations)
    # [batch_size*question_len*num_question_neighbors, compress_vector_dim]

    passage_neighbor_representations = tf.reshape(passage_neighbor_representations, [-1, input_dim + edge_dim])
    passage_neighbor_representations = tf.matmul(passage_neighbor_representations, w_compress) + b_compress
    passage_neighbor_representations = tf.tanh(passage_neighbor_representations)
    # [batch_size*passage_len*num_passage_neighbors, compress_vector_dim]
                
    # assume each node has a neighbor vector, and it is None at the beginning
    question_node_hidden = tf.zeros([batch_size, question_len, neighbor_vector_dim])
    question_node_cell = tf.zeros([batch_size, question_len, neighbor_vector_dim])

    passage_node_hidden = tf.zeros([batch_size, passage_len, neighbor_vector_dim])
    passage_node_cell = tf.zeros([batch_size, passage_len, neighbor_vector_dim])
        
        
    w_ingate = tf.get_variable("w_ingate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_ingate = tf.get_variable("u_ingate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_ingate = tf.get_variable("b_ingate", [neighbor_vector_dim], dtype=tf.float32)

    w_forgetgate = tf.get_variable("w_forgetgate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_forgetgate = tf.get_variable("u_forgetgate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_forgetgate = tf.get_variable("b_forgetgate", [neighbor_vector_dim], dtype=tf.float32)

    w_outgate = tf.get_variable("w_outgate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_outgate = tf.get_variable("u_outgate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_outgate = tf.get_variable("b_outgate", [neighbor_vector_dim], dtype=tf.float32)

    w_cell = tf.get_variable("w_cell", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_cell = tf.get_variable("u_cell", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_cell = tf.get_variable("b_cell", [neighbor_vector_dim], dtype=tf.float32)

    for i in xrange(num_syntax_match_layer):
        with tf.variable_scope('syntax_match_layer-{}'.format(i)):
            # ========for question============
            question_edge_prev_hidden = collect_neighbor_node_representations(question_node_hidden, question_neighbor_indices)
            # [batch_size, question_len, num_question_neighbors, neighbor_vector_dim]
            question_edge_prev_cell = collect_neighbor_node_representations(question_node_cell, question_neighbor_indices)
            # [batch_size, question_len, num_question_neighbors, neighbor_vector_dim]
            question_edge_prev_hidden = tf.reshape(question_edge_prev_hidden, [-1, neighbor_vector_dim])
            question_edge_prev_cell = tf.reshape(question_edge_prev_cell, [-1, neighbor_vector_dim])

            question_edge_ingate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_ingate) 
                                          + tf.matmul(question_edge_prev_hidden, u_ingate) + b_ingate)
            question_edge_forgetgate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_forgetgate) 
                                          + tf.matmul(question_edge_prev_hidden, u_forgetgate) + b_forgetgate)
            question_edge_outgate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_outgate) 
                                          + tf.matmul(question_edge_prev_hidden, u_outgate) + b_outgate)
            question_edge_cell_input = tf.tanh(tf.matmul(question_neighbor_representations,w_cell) 
                                          + tf.matmul(question_edge_prev_hidden, u_cell) + b_cell)
            question_edge_cell = question_edge_forgetgate * question_edge_prev_cell + question_edge_ingate * question_edge_cell_input
            question_edge_hidden = question_edge_outgate * tf.tanh(question_edge_cell)
            question_edge_cell = tf.reshape(question_edge_cell, [batch_size, question_len, num_question_neighbors, neighbor_vector_dim])
            question_edge_hidden = tf.reshape(question_edge_hidden, [batch_size, question_len, num_question_neighbors, neighbor_vector_dim])
            # edge mask
            question_edge_cell = tf.mul(question_edge_cell, tf.expand_dims(question_neighbor_mask, axis=-1))
            question_edge_hidden = tf.mul(question_edge_hidden, tf.expand_dims(question_neighbor_mask, axis=-1))
            question_node_cell = tf.reduce_sum(question_edge_cell, axis=2)
            question_node_hidden = tf.reduce_sum(question_edge_hidden, axis=2)
            #[batch_size, question_len, neighbor_vector_dim]

            # node mask
            question_node_cell = question_node_cell * tf.expand_dims(question_mask, axis=-1)
            question_node_hidden = question_node_hidden * tf.expand_dims(question_mask, axis=-1)

            # ========for passage============
            passage_edge_prev_hidden = collect_neighbor_node_representations(passage_node_hidden, passage_neighbor_indices)
            passage_edge_prev_cell = collect_neighbor_node_representations(passage_node_cell, passage_neighbor_indices)
            # [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim]
            passage_edge_prev_hidden = tf.reshape(passage_edge_prev_hidden, [-1, neighbor_vector_dim])
            passage_edge_prev_cell = tf.reshape(passage_edge_prev_cell, [-1, neighbor_vector_dim])

            passage_edge_ingate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_ingate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_ingate) + b_ingate)
            passage_edge_forgetgate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_forgetgate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_forgetgate) + b_forgetgate)
            passage_edge_outgate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_outgate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_outgate) + b_outgate)
            passage_edge_cell_input = tf.tanh(tf.matmul(passage_neighbor_representations,w_cell) 
                                          + tf.matmul(passage_edge_prev_hidden, u_cell) + b_cell)
            passage_edge_cell = passage_edge_forgetgate * passage_edge_prev_cell + passage_edge_ingate * passage_edge_cell_input
            passage_edge_hidden = passage_edge_outgate * tf.tanh(passage_edge_cell)
            passage_edge_cell = tf.reshape(passage_edge_cell, [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim])
            passage_edge_hidden = tf.reshape(passage_edge_hidden, [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim])
            # edge mask
            passage_edge_cell = tf.mul(passage_edge_cell, tf.expand_dims(passage_neighbor_mask, axis=-1))
            passage_edge_hidden = tf.mul(passage_edge_hidden, tf.expand_dims(passage_neighbor_mask, axis=-1))
            passage_node_cell = tf.reduce_sum(passage_edge_cell, axis=2)
            passage_node_hidden = tf.reduce_sum(passage_edge_hidden, axis=2)

            # node mask
            passage_node_cell = passage_node_cell * tf.expand_dims(passage_mask, axis=-1)
            passage_node_hidden = passage_node_hidden * tf.expand_dims(passage_mask, axis=-1)
                        
            #=====matching
            (node_matching_vectors, node_matching_dim) = match_utils.match_passage_with_question(
                                passage_node_hidden, None, passage_mask, question_node_hidden, None,question_mask, neighbor_vector_dim, 
                                with_full_matching=False, 
                                with_attentive_matching=with_attentive_matching, 
                                with_max_attentive_matching=with_max_attentive_matching, 
                                with_maxpooling_matching=with_maxpooling_matching, 
                                with_forward_match=True, with_backward_match=False, match_options=match_options)
            all_matching_vectors.extend(node_matching_vectors) #[batch_size, passage_len, node_matching_dim]
            all_matching_dim += node_matching_dim

    all_matching_vectors = tf.concat(2, all_matching_vectors) # [batch_size, passage_len, all_matching_dim]
    return (all_matching_vectors, all_matching_dim)

def graph_matching_for_chunk_ranking(in_question_repres, in_passage_repres, 
                question_mask, passage_mask, edge_embedding,
                question_neighbor_indices, passage_neighbor_indices,
                question_neighbor_edges, passage_neighbor_edges,
                question_neighbor_mask, passage_neighbor_mask,
                question_word_idx, candidate_node_idx, candidate_mask,
                neighbor_vector_dim, input_dim, edge_dim, num_syntax_match_layer,
                with_attentive_matching=True, with_max_attentive_matching=True,
                with_maxpooling_matching=True, with_local_attentive_matching=True,match_options=None): 

    all_matching_vectors = []
    all_matching_dim = 0
    
    
    input_shape = tf.shape(question_neighbor_indices)
    batch_size = input_shape[0]
    question_len = input_shape[1]
    num_question_neighbors = input_shape[2]

    input_shape = tf.shape(passage_neighbor_indices)
#     batch_size = input_shape[0]
    passage_len = input_shape[1]
    num_passage_neighbors = input_shape[2]
    
#     question_neighbor_mask = tf.sequence_mask(tf.reshape(question_neighbor_size, [-1]), num_question_neighbors, dtype=tf.float32)
#     question_neighbor_mask = tf.reshape(question_neighbor_mask, [batch_size, question_len, num_question_neighbors])

#     passage_neighbor_mask = tf.sequence_mask(tf.reshape(passage_neighbor_size, [-1]), num_passage_neighbors, dtype=tf.float32)
#     passage_neighbor_mask = tf.reshape(passage_neighbor_mask, [batch_size, passage_len, num_passage_neighbors])

    question_neighbor_edge_representations = tf.nn.embedding_lookup(edge_embedding, question_neighbor_edges) 
    # [batch_size, question_len, num_question_neighbors, edge_dim]
    passage_neighbor_edge_representations = tf.nn.embedding_lookup(edge_embedding, passage_neighbor_edges) 
    # [batch_size, passage_len, num_passage_neighbors, edge_dim]
    question_neighbor_node_representations = collect_neighbor_node_representations(
                                                    in_question_repres, question_neighbor_indices)
    # [batch_size, question_len, num_question_neighbors, input_dim]
    passage_neighbor_node_representations = collect_neighbor_node_representations(
                                                    in_passage_repres, passage_neighbor_indices)
    # [batch_size, passage_len, num_passage_neighbors, input_dim]
                
    question_neighbor_representations = tf.concat(3, [question_neighbor_node_representations, question_neighbor_edge_representations])
    # [batch_size, question_len, num_question_neighbors, input_dim+ edge_dim]
    passage_neighbor_representations = tf.concat(3, [passage_neighbor_node_representations, passage_neighbor_edge_representations])
    # [batch_size, passage_len, num_passage_neighbors, input_dim + edge_dim]

    # =====compress neighbor_representations 
    compress_vector_dim = neighbor_vector_dim
    w_compress = tf.get_variable("w_compress", [input_dim + edge_dim, compress_vector_dim], dtype=tf.float32)
    b_compress = tf.get_variable("b_compress", [compress_vector_dim], dtype=tf.float32)

    question_neighbor_representations = tf.reshape(question_neighbor_representations, [-1, input_dim + edge_dim])
    question_neighbor_representations = tf.matmul(question_neighbor_representations, w_compress) + b_compress
    question_neighbor_representations = tf.tanh(question_neighbor_representations)
    # [batch_size*question_len*num_question_neighbors, compress_vector_dim]

    passage_neighbor_representations = tf.reshape(passage_neighbor_representations, [-1, input_dim + edge_dim])
    passage_neighbor_representations = tf.matmul(passage_neighbor_representations, w_compress) + b_compress
    passage_neighbor_representations = tf.tanh(passage_neighbor_representations)
    # [batch_size*passage_len*num_passage_neighbors, compress_vector_dim]
                
    # assume each node has a neighbor vector, and it is None at the beginning
    question_node_hidden = tf.zeros([batch_size, question_len, neighbor_vector_dim])
    question_node_cell = tf.zeros([batch_size, question_len, neighbor_vector_dim])

    passage_node_hidden = tf.zeros([batch_size, passage_len, neighbor_vector_dim])
    passage_node_cell = tf.zeros([batch_size, passage_len, neighbor_vector_dim])
        
        
    w_ingate = tf.get_variable("w_ingate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_ingate = tf.get_variable("u_ingate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_ingate = tf.get_variable("b_ingate", [neighbor_vector_dim], dtype=tf.float32)

    w_forgetgate = tf.get_variable("w_forgetgate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_forgetgate = tf.get_variable("u_forgetgate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_forgetgate = tf.get_variable("b_forgetgate", [neighbor_vector_dim], dtype=tf.float32)

    w_outgate = tf.get_variable("w_outgate", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_outgate = tf.get_variable("u_outgate", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_outgate = tf.get_variable("b_outgate", [neighbor_vector_dim], dtype=tf.float32)

    w_cell = tf.get_variable("w_cell", [compress_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    u_cell = tf.get_variable("u_cell", [neighbor_vector_dim, neighbor_vector_dim], dtype=tf.float32)
    b_cell = tf.get_variable("b_cell", [neighbor_vector_dim], dtype=tf.float32)
    
    # calculate question graph representation
    all_question_graph_representations = []
    for i in xrange(num_syntax_match_layer):
        with tf.variable_scope('syntax_match_layer-{}'.format(i)):
            # ========for question============
            question_edge_prev_hidden = collect_neighbor_node_representations(question_node_hidden, question_neighbor_indices)
            # [batch_size, question_len, num_question_neighbors, neighbor_vector_dim]
            question_edge_prev_cell = collect_neighbor_node_representations(question_node_cell, question_neighbor_indices)
            # [batch_size, question_len, num_question_neighbors, neighbor_vector_dim]
            question_edge_prev_hidden = tf.reshape(question_edge_prev_hidden, [-1, neighbor_vector_dim])
            question_edge_prev_cell = tf.reshape(question_edge_prev_cell, [-1, neighbor_vector_dim])

            question_edge_ingate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_ingate) 
                                          + tf.matmul(question_edge_prev_hidden, u_ingate) + b_ingate)
            question_edge_forgetgate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_forgetgate) 
                                          + tf.matmul(question_edge_prev_hidden, u_forgetgate) + b_forgetgate)
            question_edge_outgate = tf.sigmoid(tf.matmul(question_neighbor_representations,w_outgate) 
                                          + tf.matmul(question_edge_prev_hidden, u_outgate) + b_outgate)
            question_edge_cell_input = tf.tanh(tf.matmul(question_neighbor_representations,w_cell) 
                                          + tf.matmul(question_edge_prev_hidden, u_cell) + b_cell)
            question_edge_cell = question_edge_forgetgate * question_edge_prev_cell + question_edge_ingate * question_edge_cell_input
            question_edge_hidden = question_edge_outgate * tf.tanh(question_edge_cell)
            question_edge_cell = tf.reshape(question_edge_cell, [batch_size, question_len, num_question_neighbors, neighbor_vector_dim])
            question_edge_hidden = tf.reshape(question_edge_hidden, [batch_size, question_len, num_question_neighbors, neighbor_vector_dim])
            # edge mask
            question_edge_cell = tf.mul(question_edge_cell, tf.expand_dims(question_neighbor_mask, axis=-1))
            question_edge_hidden = tf.mul(question_edge_hidden, tf.expand_dims(question_neighbor_mask, axis=-1))
#             question_node_cell = tf.reduce_sum(question_edge_cell, axis=2)
#             question_node_hidden = tf.reduce_sum(question_edge_hidden, axis=2)
            question_node_cell = tf.reduce_max(question_edge_cell, axis=2)
            question_node_hidden = tf.reduce_max(question_edge_hidden, axis=2) # TODO
            #[batch_size, question_len, neighbor_vector_dim]

            # node mask
            question_node_cell = question_node_cell * tf.expand_dims(question_mask, axis=-1)
            question_node_hidden = question_node_hidden * tf.expand_dims(question_mask, axis=-1)

            question_word_representation = collect_node_representations(question_node_hidden, question_word_idx)
            # [batch_size, neighbor_vector_dim]
            all_question_graph_representations.append(tf.reshape(question_word_representation, [batch_size, 1, neighbor_vector_dim]))
    all_question_graph_representations = tf.concat(1, all_question_graph_representations) # [batch_size, num_match_layer, neighbor_vector_dim]
     
    # calculate passage representation and match it with the question
    for i in xrange(num_syntax_match_layer):
        with tf.variable_scope('syntax_match_layer-{}'.format(i)):
            passage_edge_prev_hidden = collect_neighbor_node_representations(passage_node_hidden, passage_neighbor_indices)
            passage_edge_prev_cell = collect_neighbor_node_representations(passage_node_cell, passage_neighbor_indices)
            # [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim]
            passage_edge_prev_hidden = tf.reshape(passage_edge_prev_hidden, [-1, neighbor_vector_dim])
            passage_edge_prev_cell = tf.reshape(passage_edge_prev_cell, [-1, neighbor_vector_dim])

            passage_edge_ingate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_ingate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_ingate) + b_ingate)
            passage_edge_forgetgate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_forgetgate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_forgetgate) + b_forgetgate)
            passage_edge_outgate = tf.sigmoid(tf.matmul(passage_neighbor_representations,w_outgate) 
                                          + tf.matmul(passage_edge_prev_hidden, u_outgate) + b_outgate)
            passage_edge_cell_input = tf.tanh(tf.matmul(passage_neighbor_representations,w_cell) 
                                          + tf.matmul(passage_edge_prev_hidden, u_cell) + b_cell)
            passage_edge_cell = passage_edge_forgetgate * passage_edge_prev_cell + passage_edge_ingate * passage_edge_cell_input
            passage_edge_hidden = passage_edge_outgate * tf.tanh(passage_edge_cell)
            passage_edge_cell = tf.reshape(passage_edge_cell, [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim])
            passage_edge_hidden = tf.reshape(passage_edge_hidden, [batch_size, passage_len, num_passage_neighbors, neighbor_vector_dim])
            # edge mask
            passage_edge_cell = tf.mul(passage_edge_cell, tf.expand_dims(passage_neighbor_mask, axis=-1))
            passage_edge_hidden = tf.mul(passage_edge_hidden, tf.expand_dims(passage_neighbor_mask, axis=-1))
#             passage_node_cell = tf.reduce_sum(passage_edge_cell, axis=2)
#             passage_node_hidden = tf.reduce_sum(passage_edge_hidden, axis=2)
            passage_node_cell = tf.reduce_max(passage_edge_cell, axis=2)
            passage_node_hidden = tf.reduce_max(passage_edge_hidden, axis=2) # TODO

            # node mask
            passage_node_cell = passage_node_cell * tf.expand_dims(passage_mask, axis=-1)
            passage_node_hidden = passage_node_hidden * tf.expand_dims(passage_mask, axis=-1)
                        
            #=====matching
            canidate_node_representation = collect_node_representations(passage_node_hidden, candidate_node_idx)
            # [batch_size, num_candidate_nodes, neighbor_vector_dim]
            (node_matching_vectors, node_matching_dim) = match_utils.match_passage_with_question(
                                canidate_node_representation, None, candidate_mask, 
                                all_question_graph_representations, None,None, 
                                neighbor_vector_dim, with_full_matching=False, 
                                with_attentive_matching=with_attentive_matching, 
                                with_max_attentive_matching=with_max_attentive_matching, 
                                with_maxpooling_matching=with_maxpooling_matching, 
                                with_local_attentive_matching=with_local_attentive_matching,
                                with_forward_match=True, with_backward_match=False, match_options=match_options)
            all_matching_vectors.extend(node_matching_vectors) #[batch_size, num_candidate_nodes, node_matching_dim]
            all_matching_dim += node_matching_dim

    all_matching_vectors = tf.concat(2, all_matching_vectors) # [batch_size, num_candidate_nodes, all_matching_dim]
    return (all_matching_vectors, all_matching_dim)