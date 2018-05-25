from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_ops


class CovCopyAttenGen:
    def __init__(self, placeholders, options, vocab):
        self.options = options
        self.vocab = vocab
        self.cell = tf.contrib.rnn.LSTMCell(
                    options.gen_hidden_size,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=True)
        self.placeholders = placeholders

        with tf.variable_scope("embedding"), tf.device('/cpu:0'):
            self.embedding = tf.get_variable('word_embedding', trainable=(options.fix_word_vec==False),
                                    initializer=tf.constant(self.vocab.word_vecs), dtype=tf.float32)


    def attention(self, decoder_state, attention_vec_size, encoder_states, encoder_features, node_mask, v, w_c=None,
                  use_coverage=True, coverage=None):
        '''
        decoder_state: Tuple of [batch_size, gen_hidden_size]
        encoder_states: [batch_size, passage_len, encoder_dim]
        encoder_features: [batch_size,passage_len,attention_vec_size]
        node_mask: [batch_size, passage_len]
        v: [1,1, attention_vec_size]
        w_c: [1,1, attention_vec_size]
        coverage: [batch_size, passage_len]
        '''
        with variable_scope.variable_scope("Attention"):
            # Equation (11) in the paper
            state_features = linear(decoder_state, attention_vec_size, True) # [batch_size, attention_vec_size]
            state_features = tf.expand_dims(state_features, 1) # [batch_size, 1, attention_vec_size]
            all_features = encoder_features + state_features # [batch_size, passage_len, attention_vec_size]
            if use_coverage and coverage is not None:
                coverage_features = tf.expand_dims(coverage, axis=-1) * w_c # coverage_features: [batch_size, passage_len, 1]
                all_features += coverage_features
            e = tf.reduce_sum(v * tf.tanh(all_features), axis=-1) # [batch_size, passage_len]
            attn_dist = nn_ops.softmax(e) # [batch_size, passage_len]
            attn_dist *= node_mask

            if coverage is not None: # Update coverage vector
                coverage += attn_dist
            else: # first step of training
                coverage = attn_dist

            # Calculate the context vector from attn_dist and encoder_states
            # shape (batch_size, attn_size).
            context_vector = tf.reduce_sum(tf.expand_dims(attn_dist, axis=-1) * encoder_states, axis=1) # [batch_size, encoder_dim]
        return context_vector, attn_dist, coverage

    def embedding_lookup(self, inputs):
        '''
        inputs: list of [batch_size], int32
        '''
        if type(inputs) is list:
            return [tf.nn.embedding_lookup(self.embedding, x) for x in inputs]
        else:
            return tf.nn.embedding_lookup(self.embedding, inputs)

    def one_step_decoder(self, state_t_1, context_t_1, coverage_t_1, word_t, encoder_states, encoder_features,
                         node_idxs, node_mask, v, w_c, vocab):
        '''
        state_t_1: Tuple of [batch_size, gen_hidden_size]
        context_t_1: [batch_size, encoder_dim]
        coverage_t_1: [batch_size, passage_len]
        word_t: [batch_size, word_dim]
        encoder_states: [batch_size, passage_len, encoder_dim]
        encoder_features: [batch_size,attn_length,attention_vec_size]
        node_mask: [batch_size, passage_len]
        v: [1,1, attention_vec_size]
        w_c: [1,1, attention_vec_size]
        '''

        options = self.options
        x = linear([word_t, context_t_1], options.attention_vec_size, True)

        # Run the decoder RNN cell. cell_output = decoder state
        cell_output, state_t = self.cell(x, state_t_1)

        context_t, attn_dist, coverage_t = self.attention(state_t, options.attention_vec_size, encoder_states,
                                                             encoder_features, node_mask, v, w_c=w_c,
                                                             use_coverage=options.use_coverage, coverage=coverage_t_1)
        # Calculate p_gen, Equation (8)
        p_gen = None
        if options.pointer_gen:
            with tf.variable_scope('calculate_pgen'):
                p_gen = linear([context_t, state_t.c, state_t.h, x], 1, True) # [batch_size, 1]
                p_gen = tf.sigmoid(p_gen)

        # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
        # This is V[s_t, h*_t] + b in the paper
        with variable_scope.variable_scope("AttnOutputProjection"):
            output_t = linear([cell_output] + [context_t], options.gen_hidden_size, True)

        with tf.variable_scope('output_projection'):
            w = tf.get_variable('w', [options.gen_hidden_size, vocab.vocab_size+1], dtype=tf.float32)
            b = tf.get_variable('b', [vocab.vocab_size +1], dtype=tf.float32)
            # vocab_scores is the vocabulary distribution before applying softmax.
            # Each entry on the list corresponds to one decoder step
            vocab_score_t = tf.nn.xw_plus_b(output_t, w, b) # apply the linear layer
            vocab_score_t = tf.nn.softmax(vocab_score_t)

            # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
            if options.pointer_gen:
                vocab_score_t = self.merge_prob_dist_for_one_step(vocab_score_t, attn_dist, p_gen, node_idxs, node_mask)
            vocab_score_t = _clip_and_normalize(vocab_score_t, 1e-6)


        return (state_t, context_t, coverage_t, attn_dist, p_gen, vocab_score_t)

    def train_mode(self, vocab, encoder_dim, encoder_states, encoder_features, node_idxs, node_mask,
            init_state, decoder_inputs, answer_batch, loss_weights, mode_gen='ce_loss'):
        '''
        encoder_dim: int-valued
        encoder_states: [batch_size, passage_len, encoder_dim], for calculating context vector
        encoder_features: [batch_size, passage_len, feature_dim], the encoder part for calculating attention (W*s)
        node_idxs: [batch_size, passage_len] int32
        node_mask: [batch_size, passage_len] 0/1
        init_state: Tuple of [batch_size, gen_hidden_size]
        decoder_inputs: [batch_size, max_dec_steps].
        answer_batch: [batch_size, max_dec_steps]
        '''
        options = self.options

        input_shape = tf.shape(encoder_states)
        batch_size = input_shape[0]
        passage_len = input_shape[1]

        # map decoder inputs to word embeddings
        decoder_inputs = tf.unstack(decoder_inputs, axis=1) # max_enc_steps * [batch_size]
        answer_batch_unstack = tf.unstack(answer_batch, axis=1)

        # initialize all the variables
        state_t_1 = init_state
        context_t_1 = tf.zeros([batch_size, encoder_dim])
        coverage_t_1 = None

        # store variables from each time-step
        coverages = []
        attn_dists = []
        p_gens = []
        vocab_scores = []
        sampled_words = []
        self.encoder_features = encoder_features
        with variable_scope.variable_scope("attention_decoder"):
            # Get the weight vectors v and W_c (W_c is for coverage)
            v = variable_scope.get_variable("v", [options.attention_vec_size])
            v = tf.expand_dims(tf.expand_dims(v, axis=0), axis=0)
            w_c = None
            if options.use_coverage:
                with variable_scope.variable_scope("coverage"):
                    w_c = variable_scope.get_variable("w_c", [options.attention_vec_size])
                    w_c = tf.expand_dims(tf.expand_dims(w_c, axis=0), axis=0)

            # For each step, dec_input => lstm_output => vocab_score
            wordidx_t = decoder_inputs[0] # [batch_size] int32
            for i in range(options.max_answer_len):
                # the wordidx_t comes from the input seq for the two modes
                if mode_gen in ('ce_loss', 'rl_loss',):
                    wordidx_t = decoder_inputs[i]
                word_t = self.embedding_lookup(wordidx_t)
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                (state_t, context_t, coverage_t, attn_dist_t, p_gen_t, output_t) = self.one_step_decoder(
                                state_t_1, context_t_1, coverage_t_1, word_t, encoder_states, self.encoder_features,
                                node_idxs, node_mask, v, w_c, vocab)
                coverages.append(coverage_t)
                attn_dists.append(attn_dist_t)
                p_gens.append(p_gen_t)
                vocab_scores.append(output_t) # The vocabulary distributions.

                state_t_1 = state_t
                context_t_1 = context_t
                coverage_t_1 = coverage_t

                if mode_gen == 'greedy':
                    wordidx_t = tf.argmax(output_t, 1) # [batch_size, 1]
                    wordidx_t = tf.reshape(wordidx_t, [-1]) # [batch_size]
                elif mode_gen == 'sample':
                    log_score_t = tf.log(output_t) # [batch_size, vsize]
                    wordidx_t = tf.multinomial(log_score_t, 1) # [batch_size, 1]
                    wordidx_t = tf.reshape(wordidx_t, [-1]) # [batch_size]
                elif mode_gen in ('ce_loss', 'rl_loss',):
                    wordidx_t = answer_batch_unstack[i]
                else:
                    assert False, 'unknown generating mode %s' % mode_gen
                sampled_words.append(wordidx_t)

        if len(sampled_words)!=0:
            sampled_words = tf.stack(sampled_words, axis=1) # [batch_size, max_dec_steps]

        vocab_scores = tf.stack(vocab_scores, axis=1) # [batch_size, max_dec_steps, vocab]
        # calculating loss
        self._loss = None
        if mode_gen in ('ce_loss', 'rl_loss', ):
            xent = CE_loss(vocab_scores, answer_batch, loss_weights) # [batch_size]
            if mode_gen == 'rl_loss': xent *= self.placeholders.reward # multiply with rewards
            self._loss = tf.reduce_mean(xent)
            # Calculate coverage loss from the attention distributions
            if options.use_coverage:
                with tf.variable_scope('coverage_loss'):
                    self._coverage_loss = _coverage_loss(attn_dists, loss_weights)
                self._loss = self._loss + options.cov_loss_wt * self._coverage_loss

        # accuracy is calculated only under 'ce_loss' mode, where true answer is given
        if mode_gen == 'ce_loss':
            # The answer_batch in rl_train is actually sampled seq, not gold seq.
            # So accuracy is calculated only in 'ce_loss' mode
            accuracy = _mask_and_accuracy(vocab_scores, answer_batch, loss_weights)
            return accuracy, self._loss, sampled_words
        else:
            return None, self._loss, sampled_words

    def calculate_encoder_features(self, encoder_states, encoder_dim):
        options = self.options
        input_shape = tf.shape(encoder_states)
        batch_size = input_shape[0]
        passage_len = input_shape[1]

        with variable_scope.variable_scope("attention_decoder"):
            encoder_features = tf.expand_dims(encoder_states, axis=2) # now is shape [batch_size, passage_len, 1, encoder_dim]
            W_h = variable_scope.get_variable("W_h", [1, 1, encoder_dim, options.attention_vec_size])
            self.W_h = W_h
            encoder_features = nn_ops.conv2d(encoder_features, W_h, [1, 1, 1, 1], "SAME") # [batch_size, passage_len, 1, attention_vec_size]
            encoder_features = tf.reshape(encoder_features, [batch_size, passage_len, options.attention_vec_size])
        return encoder_features

    def decode_mode(self, word_vocab, beam_size, state_t_1, context_t_1, coverage_t_1, word_t,
                    encoder_states, encoder_features, node_idxs, node_mask):
        options = self.options

        with variable_scope.variable_scope("attention_decoder"):
            v = variable_scope.get_variable("v", [options.attention_vec_size])
            v = tf.expand_dims(tf.expand_dims(v, axis=0), axis=0)
            w_c = None
            if options.use_coverage:
                with variable_scope.variable_scope("coverage"):
                    w_c = variable_scope.get_variable("w_c", [options.attention_vec_size])
                    w_c = tf.expand_dims(tf.expand_dims(w_c, axis=0), axis=0)

            word_t_representation = self.embedding_lookup(word_t)

            (state_t, context_t, coverage_t, attn_dist_t, p_gen_t, output_t) = self.one_step_decoder(
                                state_t_1, context_t_1, coverage_t_1, word_t_representation, encoder_states, encoder_features,
                                node_idxs, node_mask, v, w_c, word_vocab)
            vocab_scores = tf.log(output_t)
            greedy_prediction = tf.reshape(tf.argmax(output_t, 1),[-1]) # calcualte greedy
            multinomial_prediction = tf.reshape(tf.multinomial(vocab_scores, 1),[-1]) # calculate multinomial
            topk_log_probs, topk_ids = tf.nn.top_k(vocab_scores, beam_size) # calculate topK
        return (state_t, context_t, coverage_t, attn_dist_t, p_gen_t, output_t, topk_log_probs, topk_ids,
                greedy_prediction, multinomial_prediction)




    def merge_prob_dist_for_one_step(self, vocab_dist, attn_dist, p_gen, node_idxs, node_mask=None):
        '''
        vocab_dist: [batch_size, vsize]
        attn_dist: [batch_size, passage_length]
        p_gen: [batch_size, 1]
        node_idxs: [batch_size, passage_length]
        node_mask: [batch_size, passage_length]
        '''
        input_shape = tf.shape(vocab_dist)
        batch_size = input_shape[0]
        vsize = input_shape[1]
        passage_length = tf.shape(node_idxs)[1]

        with tf.variable_scope('final_distribution'):
            vocab_dist = p_gen * vocab_dist
            attn_dist = (1.0-p_gen) * attn_dist

            # match attn_dist[batch_size, passage_length] to sparse one-hot representation [batch_size, passage_length, vsize]
            batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, axis=1) # shape (batch_size, 1)
            batch_nums = tf.tile(batch_nums, [1, passage_length]) # shape (batch_size, passage_length)
            step_nums = tf.range(0, limit=passage_length) # [passage_length]
            step_nums = tf.expand_dims(step_nums, axis=0) # shape (1, passage_length)
            step_nums = tf.tile(step_nums, [batch_size, 1]) # shape (batch_size, passage_length)
            indices = tf.stack((batch_nums, step_nums, node_idxs), axis=2) # shape (batch_size, passage_length, 3)
            indices = tf.reshape(indices, [-1, 3]) #[batch_size * passage_length, 3]
            indices = tf.cast(indices, tf.int64)

            shape = [batch_size, passage_length, vsize]
            shape = tf.cast(shape, tf.int64)

            attn_dist = tf.reshape(attn_dist, shape=[-1]) # [batch_size*passage_length]
            one_hot_spare_rep = tf.SparseTensor(indices=indices, values=attn_dist, dense_shape=shape) # [batch_size, passage_length, vsize]

            if node_mask is not None:
                node_mask = tf.expand_dims(node_mask, axis=-1)
                one_hot_spare_rep = one_hot_spare_rep * node_mask

            one_hot_spare_rep = tf.sparse_reduce_sum(one_hot_spare_rep, axis=1) # [batch_size, vsize]
            vocab_dist = tf.add(vocab_dist, one_hot_spare_rep)

        return vocab_dist # [batch_size, vsize]

def linear(args, output_size, bias=True, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(values=args, axis=1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term

def _clip_and_normalize(word_probs, epsilon):
    '''
    word_probs: 1D tensor of [vsize]
    '''
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True) # scale preds so that the class probas of each sample sum to 1


def CE_loss(word_probs, answers, loss_weights):
    '''
    word_probs: [batch_size, max_dec_steps, vocab]
    answers: [batch_size, max_dec_steps]
    loss_weigts: [batch_size, max_dec_steps]
    '''
    #word_probs = tf.nn.softmax(word_probs, dim=-1)
    input_shape = tf.shape(word_probs)
    vsize = input_shape[2]

    epsilon = 1.0e-6
    word_probs = _clip_and_normalize(word_probs, epsilon)

    one_hot_spare_rep = tf.one_hot(answers, vsize)

    xent = -tf.reduce_sum(one_hot_spare_rep * tf.log(word_probs), axis=-1) # [batch_size, max_dec_steps]
    if loss_weights != None:
        xent = xent * loss_weights
    xent = tf.reduce_sum(xent, axis=-1)
    return xent #[batch_size]

def _mask_and_avg(values, loss_weights):
    """Applies mask to values then returns overall average (a scalar)

      Args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        loss_weights: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

      Returns:
        a scalar
    """
    if loss_weights == None:
        return tf.reduce_mean(tf.stack(values, axis=0))

    dec_lens = tf.reduce_sum(loss_weights, axis=1) # shape batch_size. float32
    values_per_step = [v * loss_weights[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, loss_weights):
    """Calculates the coverage loss from the attention distributions.

      Args:
        attn_dists: The attention distributions for each decoder timestep.
               A list length max_dec_steps containing shape (batch_size, attn_length)
        loss_weights: shape (batch_size, max_dec_steps).

      Returns:
        coverage_loss: scalar
      """
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, loss_weights)
    return coverage_loss

# values: [batch_size, step_size, vocab_size]
# answers: [batch_size, step_size]
def _mask_and_accuracy(values, answers, loss_weights):
    values = tf.argmax(values,axis=2)
    x = tf.cast(values, dtype=tf.int32)
    y = tf.cast(answers, dtype=tf.int32)
    res = tf.equal(x, y)
    res = tf.cast(res, dtype=tf.float32)
    res = tf.multiply(res, loss_weights)
    return tf.reduce_sum(res)


