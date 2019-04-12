from models.ops import *
import tensorflow as tf

#visual features: (None, 36, 2048)
#qeustion: (None, 14, embed)
#question embedding: (None, embed_dims)

class Attention():
    def name(self):
        return "Top-Down Attention"

    def __init__(self,
                q_embed,
                v_feat,
                hidden_num=50,
                dropout=0.2,
                wd=None,
                initializer=None,
                reuse=None,
                name='attention'):
        if reuse == None:
            reuse = tf.AUTO_REUSE

        v_shape = v_feat.get_shape().as_list()
        with tf.variable_scope(name, reuse=reuse) as scope:
            q_embed_tile = tf.tile(tf.expand_dims(q_embed, axis=1), [1, v_shape[1], 1])
            #(None, 1, 1024) --> (None, 36, 1024)
            vq_concat = tf.concat([v_feat, q_embed_tile], axis=-1)
            #(None, 36, 2048), (None, 36, 3072)
            self.attention_scores(vq_concat, hidden_num, wd=wd, reuse=reuse)
            att_v_feat = tf.multiply(v_feat, self.att_scores, name='attention_visual_feature')
            self.att_v_feat = tf.reduce_sum(att_v_feat, axis=1)
            #(None, 36, 2048) * (None, 36, 1) --> (None, 36, 2048) --> (None, 2048)

    def attention_scores(self,
                        input,
                        output_dim,
                        activation='relu',
                        wd=None,
                        initializer=None,
                        reuse=None,
                        name='fc'):
        input_dim = input.get_shape()
        #(None, 36, 3072)
        input_num = [tf.shape(input)[0], 1, 1]
        #(None, 1, 1)
        with tf.variable_scope(name, reuse=reuse) as scope:
            w = get_variable(name='w', shape=[1, input_dim[-1], output_dim], initializer=initializer, wd=wd)
            w_tile = tf.tile(w, input_num)
            #(1, 3072, 1024)
            result = tf.matmul(input, w_tile)
            #(None, 36, 3072) * (None, 3072, 1024) --> #(None, 36, 1024)
            result = get_activation(result, activation)
            w_logit = get_variable(name='w_logit', shape=[1, output_dim, 1], initializer=initializer, wd=wd)
            w_logit_tile = tf.tile(w_logit, input_num)
            self.att_logit = tf.matmul(result, w_logit_tile)
            #(None, 36, 1)
            self.att_scores = softmax(self.att_logit, axis=1, name='attention_score')
             
if __name__ == "__main__":
    q_embed = tf.placeholder(tf.float32, shape=[None, 512], name='q')
    v_feat = tf.placeholder(tf.float32, shape=[None, 36, 2048], name='v')
    att = Attention(q_embed, v_feat)
    print('Attention Score Shape: ', att.att_scores)
    print('Attented Visual Feature Shape: ', att.att_v_feat)
