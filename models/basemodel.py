from models.ops import *
import tensorflow as tf
import os

class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def __init__(self, sess, args):
        self.args = args
        self.sess = sess
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.build_model()
        self.create_loss()
        self.create_summary()
        self.create_optimizer()
        self.saver = tf.train.Saver(max_to_keep=3, name='train_saver')
        self.eval_saver = tf.train.Saver(max_to_keep=1, name='eval_saver')

    def create_input_placeholder(self):
        pass

    def build_model(self):
        print("[!] Model architecture building is started")

    def run_opt(self, feed_dict):
        _, summary, loss, step = self.sess.run([self.opt, self.summary_op, self.loss, self.global_step], feed_dict=feed_dict)
        self.writer.add_summary(summary, step)
        return [loss, step]

    def run_output(self, feed_dict, logits=True):
        if logits:
            return self.sess.run(self.logits, feed_dict=feed_dict)
        else:
            return self.sess.run(self.softmax, feed_dict=feed_dict)

    def create_loss(self):
        with tf.name_scope('loss') as scope:
            self.loss = None

    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        capped_grads_and_vars = [(tf.clip_by_norm(grad, 0.25), var) for grad, var in grads_and_vars]
        opt = self.optimizer.apply_gradients(capped_grads_and_vars, global_step=self.global_step)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.opt = tf.group([opt, update_ops])

    def create_summary(self):
        pass

    def save_weights(self, checkpoint_dir, step, name=None):
        if name is None:
            model_name = self.args.model_name
        else:
            model_name = name
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step,
                        write_meta_graph=False)

    def load_weights(self, checkpoint_dir):
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            try:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Success to read {}".format(ckpt_name))
                return True
            except Exception as e:
                print("[!] Can't load: ", ckpt_name)
        else:
            print("[*] There is no saved file.")
            print("[*] Training Start from the scratch.")
            return False
