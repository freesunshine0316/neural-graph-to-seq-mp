import tensorflow as tf

def collect_representation(representation, positions):
    '''
        representation: [batch_size, passsage_length, dim]
        positions: [batch_size, num_positions]
    '''
    def singel_instance(x):
        # x[0]: [passage_length, dim]
        # x[1]: [num_positions]
        return tf.gather(x[0], x[1])
    elems = (representation, positions)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, num_positions, dim]

class PhraseProjectionLayer(object):
    def __init__(self, placeholders):
        # placeholder assignments
        self.max_phrase_size = placeholders.max_phrase_size # a scaler, max number of phrases within a batch
        self.phrase_starts = placeholders.phrase_starts # [batch_size, chunk_len]
        self.phrase_ends = placeholders.phrase_ends # [batch_size, chunk_len]
        self.phrase_lengths = placeholders.phrase_lengths # [batch_size]
    
    def project_to_phrase_representation(self, encoder_representations):
        '''
        encoder_represenations: [batch_size, passage_length, encoder_dim]
        '''
        start_representations = collect_representation(encoder_representations, self.phrase_starts) # [batch_size, chunk_len, encoder_dim]
        end_representations = collect_representation(encoder_representations, self.phrase_ends) # [batch_size, chunk_len, encoder_dim]
        phrase_representations = tf.concat(2, [start_representations, end_representations], name='phrase_representation')

        phrase_len = tf.shape(self.phrase_starts)[1]
        phrase_mask = tf.sequence_mask(self.phrase_lengths, phrase_len, dtype=tf.float32) # [batch_size, phrase_len]
        phrase_mask = tf.expand_dims(phrase_mask, axis=-1, name='phrase_mask') # [batch_size, phrase_len, 'x']
        
        phrase_representations =  phrase_representations * phrase_mask
        return  phrase_representations # [batch_size, phrase_len, 2*encoder_dim]
 


