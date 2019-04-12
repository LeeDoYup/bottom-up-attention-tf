from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import tensorflow as tf

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAEntries:
    '''
    [USE TWO ELEMENTS]
    - self.q_token: question token indexes
    - self.target: soft-score of answer
    '''
    def __init__(self, name, dictionary, dataroot='data'):
        assert name in ['train', 'val']
        self.dataroot = dataroot

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))
        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.entries = utils.convert_entries(self.entries)
        self.q_token = self.entries['q_token']
        self.images = self.entries['image']

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.target = np.zeros([len(self.entries), self.num_ans_candidates], dtype=np.float32)
        for idx, entry in enumerate(self.entries):
            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
                self.target[idx, labels] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
    
    def __entries__(self):
        return self.entries


class VQADataset:
    def __init__(self, batch, name, entries, sess, epoch=None, dataroot='data', init_type='handle'):
        self.init_type = init_type
        self.name = name
        self.batch = batch 
        self.entries = entries
        self.epoch = epoch
        self.dataroot = dataroot
        self.build_dataset(sess)
        self.initialize_dataset(sess)

    def build_dataset(self, sess):
        assert self.name in ['train', 'val']
        #if ood, additional implementations are needed
        '''        
        h5py_path = utils.get_h5py_path(self.dataroot, self.name)
        with h5py.File(h5py_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
        '''
        self.input_tensor = {}
        with tf.name_scope(self.name+'_dataset') as scope:
            self.input_tensor['images'] = tf.placeholder(self.entries.images.dtype, self.entries.images.shape, name='image_'+self.name)
            #self.input_tensor['features'] = tf.placeholder(self.features.dtype, self.features.shape, name='features_'+self.name)
            self.input_tensor['features'] = tf.placeholder(tf.float32, [None, 36, 2048], name='features_'+self.name)
            self.input_tensor['q_token'] = tf.placeholder(self.entries.q_token.dtype, self.entries.q_token.shape, name='q_token_'+self.name)
            self.input_tensor['target'] = tf.placeholder(self.entries.target.dtype, self.entries.target.shape, name='target_'+self.name)

            dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor['images'],  self.input_tensor['q_token'], self.input_tensor['target']))
            
            if self.name == 'train':
                dataset = dataset.shuffle(buffer_size=len(self.entries.target))
                dataset = dataset.repeat(self.epoch)

            def get_by_idx(images, q_token, target):
                idx = tf.reshape(images, [1])
                features = tf.gather_nd(self.input_tensor['features'], idx)
                return features,  q_token, target

            dataset = dataset.map(get_by_idx)
            dataset = dataset.batch(self.batch)
            dataset = dataset.prefetch(buffer_size=self.batch)
            self.dataset = dataset
            self.iterator = dataset.make_initializable_iterator()
            self.init = self.iterator.initializer

            self.handle = None

    def initialize_dataset(self, sess):
        h5py_path = utils.get_h5py_path(self.dataroot, self.name)
        with h5py.File(h5py_path, 'r') as hf:
            features = np.array(hf.get('image_features'))
        feed_dict = {self.input_tensor['images']: self.entries.images,
                    self.input_tensor['features']: features,
                    self.input_tensor['q_token']: self.entries.q_token,
                    self.input_tensor['target']: self.entries.target}
        sess.run(self.init, feed_dict=feed_dict)
        self.handle = sess.run(self.iterator.string_handle())