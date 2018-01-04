import tensorflow as tf
from tensorflow.python.ops import rnn

eps = 1e-6
def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def euclidean_distance(y1,y2):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance

def match_same_row_matrix(matrix1, matrix2, feature_dim, options):
    # matrix1: [num_rows, feature_dim]
    # matrix2: [num_rows, feature_dim]
    input_shape = tf.shape(matrix1)
    num_rows = input_shape[0]
    matching_result = []
    matching_dim = 0
    if options.has_key('cosine'):
        cosine_value = cosine_distance(matrix1, matrix2)
        cosine_value = tf.reshape(cosine_value, [num_rows, 1])
        matching_result.append(cosine_value)
        matching_dim += 1

    if options.has_key('euclidean'):
        euclidean_value = euclidean_distance(matrix1, matrix2)
        euclidean_value = tf.reshape(euclidean_value, [num_rows, 1])
        matching_result.append(euclidean_value)
        matching_dim += 1

    if options.has_key('subtract'):
        cur_matching = matrix1-matrix2
#         cur_matching = tf.reshape(cur_matching, [num_rows, feature_dim])
        matching_result.append(cur_matching)
        matching_dim += feature_dim

    if options.has_key('multiply'):
        cur_matching = matrix1*matrix2
#         cur_matching = tf.reshape(cur_matching, [num_rows, feature_dim])
        matching_result.append(cur_matching)
        matching_dim += feature_dim

    if options.has_key('nn'):
        (nn_dim, w, b) = options['nn']
#         with tf.variable_scope(name_scope + "-nn"):
#             w = tf.get_variable("w", [2*feature_dim, nn_dim], dtype=tf.float32)
#             b = tf.get_variable("b", [nn_dim], dtype=tf.float32)
        NN_input = tf.concat(1, [matrix1, matrix2])
        NN_input = tf.reshape(NN_input, [num_rows, 2*feature_dim])
        NN_matching = tf.tanh(tf.matmul(NN_input, w) + b)
        NN_matching = tf.reshape(NN_matching, [num_rows, nn_dim])
        matching_result.append(NN_matching)
        matching_dim += nn_dim

    matrix1_tmp = tf.expand_dims(matrix1, axis=1) #[num_rows, 'x', feature_dim]
    matrix2_tmp = tf.expand_dims(matrix2, axis=1) #[num_rows, 'x', feature_dim]
    if options.has_key('mp-cosine'):
        (cosine_MP_dim, mp_cosine_params) = options['mp-cosine']
#         mp_cosine_params = tf.get_variable(name_scope + "-mp-cosine", shape=[cosine_MP_dim, feature_dim], dtype=tf.float32)
        mp_cosine_params_tmp = tf.expand_dims(mp_cosine_params, axis=0) # ['x', cosine_MP_dim, feature_dim]
        mp_cosine_matching = cosine_distance(tf.multiply(matrix1_tmp, mp_cosine_params_tmp), tf.multiply(matrix2_tmp, mp_cosine_params_tmp))
#         mp_cosine_matching = tf.reshape(mp_cosine_matching, [num_rows, cosine_MP_dim])
        matching_result.append(mp_cosine_matching)
        matching_dim += cosine_MP_dim

    if options.has_key('mp-euclidean'):
        (euclidean_MP_dim, mp_euclidean_params) = options['mp-euclidean']
#         mp_euclidean_params = tf.get_variable(name_scope + "-mp-euclidean", shape=[euclidean_MP_dim, feature_dim], dtype=tf.float32)
        mp_euclidean_params_tmp = tf.expand_dims(mp_euclidean_params, axis=0) # ['x', euclidean_MP_dim, feature_dim]
        mp_euclidean_matching = euclidean_distance(tf.multiply(matrix1_tmp, mp_euclidean_params_tmp), tf.multiply(matrix2_tmp, mp_euclidean_params_tmp))
#         mp_euclidean_matching = tf.reshape(mp_euclidean_matching, [num_rows, euclidean_MP_dim])
        matching_result.append(mp_euclidean_matching)
        matching_dim += euclidean_MP_dim

    matching_result = tf.concat(1, matching_result)
    return (matching_result, matching_dim)

def match_matrix_bak(matrix1, matrix2, feature_dim, options):
    # matrix1: [num_rows1, feature_dim]
    # matrix2: [num_rows2, feature_dim]
    num_rows1 = tf.shape(matrix1)[0]
    num_rows2 = tf.shape(matrix2)[0]

    matrix1_tmp = tf.expand_dims(matrix1, axis=1) # [num_rows1, 'x', feature_dim]
    matrix1_tmp = tf.tile(matrix1_tmp, [1, num_rows2, 1], name=None)# [num_rows1, num_rows2, feature_dim]
    matrix1_tmp = tf.reshape(matrix1_tmp, [num_rows1*num_rows2, feature_dim])

    matrix2_tmp = tf.expand_dims(matrix2, axis=0) # ['x', num_rows1, feature_dim]
    matrix2_tmp = tf.tile(matrix2_tmp, [num_rows1, 1, 1], name=None)# [num_rows1, num_rows2, feature_dim]
    matrix2_tmp = tf.reshape(matrix2_tmp, [num_rows1*num_rows2, feature_dim])

    (matching_result, matching_dim) = match_same_row_matrix(matrix1_tmp, matrix2_tmp, feature_dim, options)
    matching_result = tf.reshape(matching_result, [num_rows1, num_rows2, matching_dim])
    return (matching_result, matching_dim)

def match_matrix_bak2(matrix1, matrix2, feature_dim, options):
    # matrix1: [num_rows1, feature_dim]
    # matrix2: [num_rows2, feature_dim]
    num_rows2 = tf.shape(matrix2)[0]
    def singel_instance(x):
        # x: [feature_dim]
        x = tf.reshape(x, [1, feature_dim]) # ['x', feature_dim]
        x = tf.tile(x, [num_rows2, 1])# [num_rows2, feature_dim]
        (cur_matching_result, _) = match_same_row_matrix(x, matrix2, feature_dim, options)
        return cur_matching_result
    matching_result = tf.map_fn(singel_instance, matrix1, dtype=tf.float32) # [num_rows1, num_rows2, matching_dim]
    matching_dim = options['matching_dim']
    return (matching_result, matching_dim)

def match_matrix(matrix1, matrix2, feature_dim, options):
    # matrix1: [num_rows1, feature_dim]
    # matrix2: [num_rows2, feature_dim]
    num_rows1 = tf.shape(matrix1)[0]
    num_rows2 = tf.shape(matrix2)[0]
    def singel_instance(x):
        # x: [feature_dim]
        x = tf.reshape(x, [1, feature_dim]) # ['x', feature_dim]
        def single_word(y):
            # y: {features_dim}
            y = tf.reshape(y, [1, feature_dim]) # ['x', feature_dim]
            (cur_matching_result, _) = match_same_row_matrix(x, y, feature_dim, options)
            return cur_matching_result
        return tf.map_fn(single_word, matrix2, dtype=tf.float32) # [num_rows1, num_rows2, matching_dim]
    matching_dim = options['matching_dim']
    matching_result = tf.map_fn(singel_instance, matrix1, dtype=tf.float32) # [num_rows1, num_rows2, matching_dim]
    matching_result = tf.reshape(matching_result, [num_rows1, num_rows2, matching_dim])
    return (matching_result, matching_dim)


def create_matching_params(feature_dim, options, name_scope):
    options_with_params = {}
    matching_dim = 0
    if options.with_cosine:
        options_with_params['cosine'] = 'cosine'
        matching_dim += 1

    if options.with_euclidean:
        options_with_params['euclidean'] = 'euclidean'
        matching_dim += 1

    if options.with_subtract:
        options_with_params['subtract'] = 'subtract'
        matching_dim += feature_dim

    if options.with_multiply:
        options_with_params['multiply'] = 'multiply'
        matching_dim += feature_dim

    if options.with_nn_match:
        nn_dim = options.nn_match_dim
        with tf.variable_scope(name_scope + "-nn"):
            w = tf.get_variable("w_nn_match", [2*feature_dim, nn_dim], dtype=tf.float32)
            b = tf.get_variable("b_nn_match", [nn_dim], dtype=tf.float32)
        options_with_params['nn'] = (nn_dim, w, b)
        matching_dim += nn_dim

    if options.with_mp_cosine:
        cosine_MP_dim = options.cosine_MP_dim
        mp_cosine_params = tf.get_variable(name_scope + "-mp-cosine", shape=[cosine_MP_dim, feature_dim], dtype=tf.float32)
        options_with_params['mp-cosine'] = (cosine_MP_dim, mp_cosine_params)
        matching_dim += cosine_MP_dim

    if options.with_mp_euclidean:
        euclidean_MP_dim = options.euclidean_MP_dim
        mp_euclidean_params = tf.get_variable(name_scope + "-mp-euclidean", shape=[euclidean_MP_dim, feature_dim], dtype=tf.float32)
        options_with_params['mp-euclidean'] = (euclidean_MP_dim, mp_euclidean_params)
        matching_dim += euclidean_MP_dim
    options_with_params['matching_dim'] = matching_dim
    return options_with_params

def calculate_full_matching(passage_representation, full_question_representation, feature_dim, options, name_scope):
    # passage_representation: [batch_size, passage_len, feature_dim]
    # full_question_representation: [batch_size, feature_dim]

    # create parameters
    options_with_params = create_matching_params(feature_dim, options, name_scope)
    matching_dim = options_with_params['matching_dim']

    in_shape = tf.shape(passage_representation)
    batch_size = in_shape[0]
    passage_len = in_shape[1]

    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, feature_dim], q: [feature_dim]
        q = tf.expand_dims(q, axis=0) # ['x', feature_dim]
        q = tf.tile(q, [passage_len, 1])# [pasasge_len, feature_dim]
        (cur_matching_result, _) = match_same_row_matrix(p, q, feature_dim, options_with_params)
        return cur_matching_result
    elems = (passage_representation, full_question_representation)
    matching_result = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]
#     matching_result = tf.reshape(matching_result, [batch_size, passage_len, matching_dim])
    return (matching_result, matching_dim)

def calculate_maxpooling_matching(passage_rep, question_rep, feature_dim, options, name_scope):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]

    # create parameters
    options_with_params = create_matching_params(feature_dim, options, name_scope)
    matching_dim = options_with_params['matching_dim']

    in_shape = tf.shape(passage_rep)
    batch_size = in_shape[0]
    passage_len = in_shape[1]

    def singel_instance(x):
        p = x[0] # p: [pasasge_len, dim]
        q = x[1] # q: [question_len, dim]
        (cur_matching_result, _) = match_matrix(p, q, feature_dim, options_with_params)
        return cur_matching_result # [pasasge_len, question_len, matching_dim]
    elems = (passage_rep, question_rep)
    matching_result = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, matching_dim]
    matching_result = tf.concat(2, [tf.reduce_max(matching_result, axis=2), tf.reduce_mean(matching_result, axis=2)])
#     matching_result = tf.reshape(matching_result, [batch_size, passage_len, 2*matching_dim])
    return (matching_result, 2*matching_dim)

def calculate_attentive_matching(passage_rep, att_question_rep, feature_dim, options, name_scope):
    # passage_rep: [batch_size, passage_len, dim]
    # att_question_rep: [batch_size, passage_len, dim]

    # create parameters
    options_with_params = create_matching_params(feature_dim, options, name_scope)
    matching_dim = options_with_params['matching_dim']

    in_shape = tf.shape(passage_rep)
    batch_size = in_shape[0]
    passage_len = in_shape[1]

    def singel_instance(x):
        p = x[0] # p: [pasasge_len, dim]
        q = x[1] # q: [pasasge_len, dim]
        (cur_matching_result, _) = match_same_row_matrix(p, q, feature_dim, options_with_params)
        return cur_matching_result

    elems = (passage_rep, att_question_rep)
    matching_result = tf.map_fn(singel_instance, elems, dtype=tf.float32)
#     matching_result = tf.reshape(matching_result, [batch_size, passage_len, matching_dim])
    return (matching_result, matching_dim)

def calculate_cosine_weighted_question_representation(question_representation, cosine_matrix, normalize=False):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, passage_len, question_len, 'x']
    weighted_question_words = tf.expand_dims(question_representation, axis=1) # [batch_size, 'x', question_len, dim]
    weighted_question_words = tf.reduce_sum(tf.multiply(weighted_question_words, expanded_cosine_matrix), axis=2)# [batch_size, passage_len, dim]
    if not normalize:
        weighted_question_words = tf.div(weighted_question_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_question_words # [batch_size, passage_len, dim]

def calculate_max_question_representation(question_representation, cosine_matrix):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    question_index = tf.arg_max(cosine_matrix, 2) # [batch_size, passage_len]
    def singel_instance(x):
        q = x[0]
        c = x[1]
        return tf.gather(q, c)
    elems = (question_representation, question_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def calculate_local_question_representation(question_representation, cosine_matrix, win_size):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    in_shape = tf.shape(question_representation)
#     batch_size = in_shape[0]
    question_len = tf.cast(in_shape[1], tf.int64)
    question_index = tf.arg_max(cosine_matrix, 2) # [batch_size, passage_len]
    def singel_instance(x):
        q = x[0] # question_representation: [question_len, dim]
        c = x[1] # question_index: [question_len]
        result = tf.gather(q, c)
        for i in xrange(win_size):
            cur_index = tf.subtract(c, i+1)
            cur_index = tf.maximum(cur_index, 0)
            result = result + tf.gather(q, cur_index)

        for i in xrange(1, win_size):
            cur_index = tf.add(c, i+1)
            cur_index = tf.minimum(cur_index, question_len-1)
            result = result + tf.gather(q, cur_index)

        return result / (2*win_size + 1)
    elems = (question_representation, question_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def cal_linear_decomposition_representation(passage_representation, passage_lengths, cosine_matrix,is_training,
                                            lex_decompsition_dim, dropout_rate):
    # passage_representation: [batch_size, passage_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    passage_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, passage_len]
    similar_weights = tf.expand_dims(passage_similarity, -1) # [batch_size, passage_len, 1]
    dissimilar_weights = tf.subtract(1.0, similar_weights)
    similar_component = tf.multiply(passage_representation, similar_weights)
    dissimilar_component = tf.multiply(passage_representation, dissimilar_weights)
    all_component = tf.concat(2, [similar_component, dissimilar_component])
    if lex_decompsition_dim==-1:
        return all_component
    with tf.variable_scope('lex_decomposition'):
        lex_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        lex_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lex_decompsition_dim)
        if is_training:
            lex_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lex_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        lex_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_fw])
        lex_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lex_lstm_cell_bw])

        (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
                    lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=passage_lengths)

        lex_features = tf.concat(2, [lex_features_fw, lex_features_bw])
    return lex_features



def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def match_passage_with_question(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                context_lstm_dim, with_full_matching=True, with_attentive_matching=True,
                                with_max_attentive_matching=True, with_maxpooling_matching=True,
                                with_forward_match=True, with_backward_match=True, match_options=None,
                                with_local_attentive_matching=False, win_size=3):
    all_question_aware_representatins = []
    dim = 0

    if with_forward_match:
        if question_mask is not None:
            question_context_representation_fw = tf.multiply(question_context_representation_fw, tf.expand_dims(question_mask,-1))
        passage_context_representation_fw = tf.multiply(passage_context_representation_fw, tf.expand_dims(mask,-1))

    if with_backward_match:
        if question_mask is not None:
            question_context_representation_bw = tf.multiply(question_context_representation_bw, tf.expand_dims(question_mask,-1))
        passage_context_representation_bw = tf.multiply(passage_context_representation_bw, tf.expand_dims(mask,-1))

    if with_full_matching:
        # forward full matching
        if with_forward_match:
            fw_question_full_rep = question_context_representation_fw[:,-1,:]
            (fw_full_match_rep, matching_dim) = calculate_full_matching(passage_context_representation_fw, fw_question_full_rep,
                                                                         context_lstm_dim, match_options, 'fw_full_match')
            all_question_aware_representatins.append(fw_full_match_rep)
            dim += matching_dim

        # backward full matching
        if with_backward_match:
            bw_question_full_rep = question_context_representation_bw[:,0,:]
            (bw_full_match_rep, matching_dim) = calculate_full_matching(passage_context_representation_bw, bw_question_full_rep,
                                                                         context_lstm_dim, match_options, 'bw_full_match')
            all_question_aware_representatins.append(bw_full_match_rep)
            dim += matching_dim

    if with_maxpooling_matching:
        # forward Maxpooling-Matching
        if with_forward_match:
            (fw_maxpooling_rep, matching_dim) = calculate_maxpooling_matching(passage_context_representation_fw,
                                        question_context_representation_fw, context_lstm_dim, match_options, 'fw_maxpooling_match')
            all_question_aware_representatins.append(fw_maxpooling_rep)
            dim += matching_dim

        # backward Maxpooling-Matching
        if with_backward_match:
            (bw_maxpooling_rep, matching_dim) = calculate_maxpooling_matching(passage_context_representation_bw,
                                    question_context_representation_bw, context_lstm_dim, match_options, 'bw_maxpooling_match')
            all_question_aware_representatins.append(bw_maxpooling_rep)
            dim += matching_dim

    if with_forward_match:
        forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)

    if with_backward_match:
        backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)

    if with_attentive_matching:
        # forward attentive-matching
        if with_forward_match:
            att_question_fw_contexts = calculate_cosine_weighted_question_representation(question_context_representation_fw,
                                                                                     forward_relevancy_matrix)
            (fw_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_fw,
                                                    att_question_fw_contexts, context_lstm_dim, match_options, 'fw_attentive_match')
            all_question_aware_representatins.append(fw_attentive_rep)
            dim += matching_dim

        # backward attentive-matching
        if with_backward_match:
            att_question_bw_contexts = calculate_cosine_weighted_question_representation(question_context_representation_bw,
                                                                                     backward_relevancy_matrix)
            (bw_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_bw,
                                                    att_question_bw_contexts, context_lstm_dim, match_options, 'bw_attentive_match')
            all_question_aware_representatins.append(bw_attentive_rep)
            dim += matching_dim

    if with_max_attentive_matching:
        # forward max attentive-matching
        if with_forward_match:
            max_att_fw = calculate_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
            (fw_max_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_fw,
                                                    max_att_fw, context_lstm_dim, match_options, 'fw_max_attentive_match')
            all_question_aware_representatins.append(fw_max_attentive_rep)
            dim += matching_dim

        # backward max attentive-matching
        if with_backward_match:
            max_att_bw = calculate_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
            (bw_max_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_bw,
                                                    max_att_bw, context_lstm_dim, match_options, 'bw_max_attentive_match')
            all_question_aware_representatins.append(bw_max_attentive_rep)
            dim += matching_dim

    if with_local_attentive_matching:
        # forward max attentive-matching
        if with_forward_match:
            local_att_fw = calculate_local_question_representation(question_context_representation_fw, forward_relevancy_matrix, win_size)
            (fw_local_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_fw,
                                                    local_att_fw, context_lstm_dim, match_options, 'fw_local_attentive_match')
            all_question_aware_representatins.append(fw_local_attentive_rep)
            dim += matching_dim

        # backward max attentive-matching
        if with_backward_match:
            local_att_bw = calculate_local_question_representation(question_context_representation_bw, backward_relevancy_matrix, win_size)
            (bw_local_attentive_rep, matching_dim) = calculate_attentive_matching(passage_context_representation_bw,
                                                    local_att_bw, context_lstm_dim, match_options, 'bw_local_attentive_match')
            all_question_aware_representatins.append(bw_local_attentive_rep)
            dim += matching_dim

    if with_forward_match:
        all_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 2

    if with_backward_match:
        all_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 2
    return (all_question_aware_representatins, dim)

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]

def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val



