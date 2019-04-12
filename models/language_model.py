import tensorflow as tf
import numpy as np 

class WordEmbedding:
    def __init__(self, filename, padding=True, name='word_embedding'):
        glove_init = np.load(filename)
        self.ntoken, self.emb_dim = np.shape(glove_init)
        self._word_embed = tf.Variable(glove_init, name=name)

        if padding:
            zero_padding = tf.zeros([1, self.emb_dim])
            self.word_embed = tf.concat([self._word_embed, zero_padding], axis=0)
        else:
            self.word_embed = self._word_embed

    def lookup(self, input):
        #shapes: input=[batch,], self.embedding=[batch, emb_dim]
        self.embedding = tf.nn.embedding_lookup(self.word_embed, input)
        return self.embedding


class QuestionEmbedding:
    def __init__(self, h_dim, n_layer=1, rnn_type='GRU', name='question_embedding'):
        self.rnn_type = rnn_type
        self.n_layer = n_layer
        if rnn_type not in ['RNN', 'LSTM', 'GRU']:
            raise NotImplementedError("NO VALID TYPE OF RNN_TYPE")
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if rnn_type == 'RNN':
                cell = tf.nn.rnn_cell.BasicRNNCell
            elif rnn_type == 'LSTM':
                cell = tf.nn.rnn_cell.LSTMCell
            else:
                cell = tf.nn.rnn_cell.GRUCell

            if n_layer ==1:
                self.cell = cell(h_dim)
            else:
                if len(h_dim) == layer:
                    self.cell = tf.nn.rnn_cell.MultiRNNCell([cell(hidden_size) for hidden_size in h_dim])
                else:
                    self.cell = tf.nn.rnn_cell.MultiRNNCell([cell(h_dim[0]) for _ in range(n_layer)])

    def build_model(self, input):
        self.output, self.state = tf.nn.dynamic_rnn(cell=self.cell, inputs=input, dtype=tf.float32)
        if self.rnn_type == 'LSTM' or self.n_layer>1:
            return self.state[-1]
        else:
            return self.state

if __name__ == '__main__':
    print('TEST YOUR CODES !!!!')
