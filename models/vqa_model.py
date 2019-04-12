import os
import sys
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) )

from models.basemodel import BaseModel
from models.top_down_attention import Attention
from models.language_model import WordEmbedding, QuestionEmbedding
from models.ops import *
from dataset import *

import tensorflow as tf
import numpy as np


class VQA_Model(BaseModel):
    def __init__(self, sess, args, entries):
        self.output_dim = entries['val'].num_ans_candidates
        self.num_hid = args.num_hid
        self.entries = entries
        self.args = args
        self.sess = sess
        self.build_dataset(self.args.mode)
        super(VQA_Model, self).__init__(sess, args)

    def name(self):
        return "Bottom-up and Top-down Attention for VQA"

    def create_input_placeholder(self):
        self.input = {'v': self.next_element[0],
                    'q': self.next_element[1],
                    'a': self.next_element[2]}
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.drop_p = tf.placeholder(tf.float32, name='drop_p')
        self.training = tf.placeholder(tf.bool, name='is_train')

    def build_dataset(self, name):
        self.dataset = {}
        if name == 'train':
            self.dataset[name] = VQADataset(self.args.batch,
                                                name, self.entries[name], self.sess, self.args.epoch)

        self.dataset['eval'] = VQADataset(self.args.batch,
                                            'val', self.entries['val'], self.sess)

        #self.next_element #[0]: features , [1]:spatial, [2]: questions, [3]: target
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='handle')
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle, 
                            self.dataset['eval'].dataset.output_types, 
                            self.dataset['eval'].dataset.output_shapes)
        self.next_element = self.iterator.get_next()

    def build_model(self):
        super(VQA_Model, self).build_model()
        self.create_input_placeholder()
        with tf.variable_scope('VQA_model', reuse=tf.AUTO_REUSE) as scope:
            self.embedding = WordEmbedding(self.args.embed_path)
            w_embed = self.embedding.lookup(self.input['q'])
            #shape = [None, 14, embed_dim]

            self.Q_embedding = QuestionEmbedding(self.num_hid)
            q_embed = self.Q_embedding.build_model(w_embed)
            self.q_embed = q_embed
            #shape = [None, question_dim]

            attention = Attention(q_embed, self.input['v'])
            v_net = fc_layer(attention.att_v_feat, self.args.num_hid, activation='relu', name='v_net')
            q_net = fc_layer(q_embed, self.args.num_hid, activation='relu', name='q_net')
            self.v_net, self.q_net = v_net, q_net

            joint_repr = tf.multiply(v_net, q_net, name='fusion')
            self.joint_repr = joint_repr
            self.logits = VQA_classifier(joint_repr, self.args.num_hid*2,
                            self.output_dim, self.drop_p, self.training, name='classifier')

            self.score_op = compute_score_with_logits(self.logits, self.input['a'], self.output_dim)
            self.score_sum = tf.reduce_sum(self.score_op)
            self.score_mean = tf.reduce_mean(tf.reduce_sum(self.score_op,axis=-1))

    def create_loss(self):
        with tf.name_scope('loss') as scope:
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            #    labels=self.input['a'], logits=self.logits, name='sample_cross_entropy_loss')
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input['a'], logits=self.logits, name='sample_cross_entropy_loss')
            self.ce_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', self.ce_loss)
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        print('[*] Losses are created')

    def create_summary(self):
        self.writer = tf.summary.FileWriter(self.args.summary_dir+'/'+self.args.model_name+'/train', self.sess.graph)
        self.eval_writer = tf.summary.FileWriter(self.args.summary_dir+'/'+self.args.model_name +'/eval' )
        summaries = [tf.summary.scalar('loss/total_loss', self.loss),
                    tf.summary.scalar('loss/cross_entropy', self.ce_loss),
                    tf.summary.scalar('score/score', self.score_mean)]
        self.summary_op = tf.summary.merge(summaries)
        print("[*] Summary Ops are created")

    def run_opt(self, feed_dict):
        _, summary, loss, total_score, step = self.sess.run([self.opt, self.summary_op, self.loss, self.score_sum, self.global_step], feed_dict=feed_dict)
        self.writer.add_summary(summary, step)
        return [loss, total_score, step]

    def run_eval(self, feed_dict, step):
        summary, loss, total_score = self.sess.run([self.summary_op, self.loss, self.score_sum], feed_dict=feed_dict)
        self.eval_writer.add_summary(summary, step)
        return [loss, total_score]

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.load_weights(self.args.checkpoint_dir)

        train_num = len(self.dataset['train'].entries.target)
        iter_count = 0
        patient = 0
        epoch = 1
        total_loss, total_score = 0.0, 0.0
        min_eval_score = 10000000.0
        feed_dict = {self.dataset_handle: self.dataset['train'].handle,
                    self.learning_rate: self.args.lr,
                    self.drop_p: self.args.drop_p,
                    self.training: True}
        print("[*] Training Strated")
        while True:
            try:
                iter_count += 1
                if patient > self.args.patient:
                    print('[!] training is early stoped\n')
                    break
                loss, score, step = self.run_opt(feed_dict=feed_dict)
                total_loss += loss
                total_score += score
                
                if int(iter_count * self.args.batch) > epoch*train_num:
                    total_loss /= float(train_num/self.args.batch)
                    total_score = total_score*100 / float(train_num)
                    #eval_loss, eval_score = self.evaluate(step)
                    #print train/eval results of each epoch]
                    print(epoch, '-th epoch\n', 
                        '(mean) train loss:\t', total_loss, ', train score:\t', total_score, '\n') 
                    epoch += 1
                    total_loss, total_score = 0.0, 0.0
                    '''
                    if (epoch > self.args.epoch/2.0):
                        if min_eval_score > eval_score:
                            min_eval_loss = eval_loss
                            self.save_weights(self.args.checkpoint_dir, 0, self.args.model_name+'_best_eval')
                            patient = 0
                        else:
                            patient += 1
                    '''
            except tf.errors.OutOfRangeError:
                break
                
        self.save_weights(self.args.checkpoint_dir, step)


    def evaluate(self, step):
        if not self.args.is_train:
            self.sess.run(tf.global_variables_initializer())
            self.load_weights(self.args.checkpoint_dir)
        total_loss, total_score = 0.0, 0.0
        self.dataset['eval'].initialize_dataset(self.sess)
        feed_dict = {self.dataset_handle: self.dataset['eval'].handle,
                    self.learning_rate: self.args.lr,
                    self.drop_p: self.args.drop_p,
                    self.training: False}
        iter_step=0
        while True:
            try:
                loss, score = self.run_eval(feed_dict=feed_dict, step=step+iter_step)
                total_loss += loss
                total_score += score
                iter_step +=1
            except tf.errors.OutOfRangeError:
                break
        eval_num = len(self.dataset['eval'].entries.target)
        eval_loss = total_loss / float(eval_num)
        eval_score = total_score*100 / float(eval_num)

        print('\n [Last Evaluation Result] \n')
        print('[*] Mean Eval Loss:\t', eval_loss)
        print('[*] Evaluation Score:\t', eval_score)
        
        return eval_loss, eval_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--L2', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--patient', type=int, default=500)
    parser.add_argument('--drop_p', type=float, default=0.2)

    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--is_train', action='store_true')
    
    parser.add_argument('--model_name', type=str, default='no_named')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_saved/')
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--summary_dir', type=str, default='./tensorboard')

    parser.add_argument('--embed_path', type=str, default= './data/glove6b_init_300d.npy')

    parser.add_argument('--num_gpu', type=int, default=1)
    args = parser.parse_args()

    path = './data/'

    try:
        is_train = args.is_train
        args.mode = 'train'
    except:
        args.is_train = False 
        args.mode = 'eval'
    
    args.is_train = True

    dictionary = Dictionary.load_from_file(path+'dictionary.pkl')
    entries = {}
    for name in ['train', 'val']:
        entries[name] = VQAEntries(name, dictionary, path)

    sess = tf.Session()
    #load and preprosess dataset
    sess = tf.Session()
    model = VQA_Model(sess, args, entries)
    model.train()
