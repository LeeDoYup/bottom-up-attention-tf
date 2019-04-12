from __future__ import print_function
from __future__ import division

from dataset import Dictionary, VQAEntries
import numpy as np 
import tensorflow as tf 
import random

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--L2', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--patient', type=int, default=500)
    parser.add_argument('--drop_p', type=float, default=0.0)

    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--is_train', action='store_true')
    
    parser.add_argument('--model_name', type=str, default='no_named')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_saved/')
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--summary_dir', type=str, default='./tensorboard')

    parser.add_argument('--embed_path', type=str, default= './data/glove6b_init_300d.npy')

    parser.add_argument('--num_gpu', type=int, default=1)
    args = parser.parse_args()

    try:
        is_train = args.is_train
        args.mode = 'train'
    except:
        args.is_train = False 
        args.mode = 'eval'

    dictionary = Dictionary.load_from_file('./data/dictionary.pkl')
    entries = {}
    for name in ['train', 'val']:
        entries[name] = VQAEntries(name, dictionary)

    #load and preprosess dataset
    if args.num_gpu < 2:
        from models.vqa_model import VQA_Model as VQA_Model
    else:
        from models.vqa_model_gpus import VQA_Model_gpus as VQA_Model

    sess = tf.Session()
    model = VQA_Model(sess, args, entries)

    if args.mode == 'train':
        print("\n [!] Training Starts")
        model.train()
        print("\n [!] Training Ends")
    else:
        model.evaluate(0)
