from models.basemodel import BaseModel
from models.top_down_attention import Attention
from models.language_model import WordEmbedding, QuestionEmbedding
from models.vqa_model import *
from models.ops import *

import tensorflow as tf
import numpy as np


class VQA_Model_gpus(VQA_Model):
    def name(self):
        return "Bottom-up and Top-down Attention for VQA, multi-gpu version"

    def create_input_placeholder(self):
        super(VQA_Model_gpus, self).create_input_placeholder()
        self.input_split = {'v': tf.split(self.input['v'], int(self.args.num_gpu), name='visual_input_split'),
                        'q': tf.split(self.input['q'], int(self.args.num_gpu), name='question_input_split'),
                        'a': tf.split(self.input['a'], int(self.args.num_gpu), name='answer_input_split')}

    def build_model(self):
        super(VQA_Model, self).build_model()
        self.create_input_placeholder()

        self.logits_list = []
        with tf.variable_scope('VQA_model', reuse=tf.AUTO_REUSE) as scope:
            self.embedding = WordEmbedding(self.args.embed_path)
            for gpu_id in range(int(self.args.num_gpu)):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    logit = self.model(gpu_id)
                    self.logits_list.append(logit)

            self.logits = tf.concat(self.logits_list, axis=0, name='logits')
            self.score_op = compute_score_with_logits(self.logits, self.input['a'], self.output_dim)
            self.score_sum = tf.reduce_sum(self.score_op)
            self.score_mean = tf.reduce_mean(self.score_op)

    def model(self, gpu_id):
            w_embed = self.embedding.lookup(self.input_split['q'][gpu_id])

            self.Q_embedding = QuestionEmbedding(self.num_hid)
            q_embed = self.Q_embedding.build_model(w_embed)

            attention = Attention(q_embed, self.input_split['v'][gpu_id])
            v_net = fc_layer(attention.att_v_feat, self.args.num_hid, activation='relu', name='v_net')
            q_net = fc_layer(q_embed, self.args.num_hid, activation='relu', name='q_net')

            joint_repr = tf.multiply(v_net, q_net, name='fusion')
            logits = VQA_classifier(joint_repr, self.args.num_hid*2,
                            self.output_dim, self.drop_p, self.training, name='classifier')
            return logits

    def model_loss(self, gpu_id):
        with tf.name_scope(str(gpu_id)+'_loss') as scope:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input_split['a'][gpu_id], logits=self.logits_list[gpu_id], name='sample_cross_entropy_loss_'+str(gpu_id))
            return cross_entropy
    
    def create_loss(self):
        with tf.name_scope('loss') as scope:
            losses = []
            for gpu_id in range(int(self.args.num_gpu)):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    loss = self.model_loss(gpu_id)
                    losses.append(loss)

            self.losses = tf.concat(losses, axis=0, name='loss_concat')
            self.ce_loss = tf.reduce_mean(self.losses, name='sample_cross_entropy_loss')
            tf.add_to_collection('losses', self.ce_loss)
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        print('[*] Losses are created')


