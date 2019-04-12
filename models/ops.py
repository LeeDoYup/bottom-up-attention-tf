import tensorflow as tf

def get_variable(name,
                  shape,
                  initializer,
                  dtype=tf.float32,
                  wd=None):
  w = tf.get_variable(name, shape, dtype=dtype, initializer=initializer)
  if wd is not None:
    add_regularization(wd, w)
  return w


def add_regularization(wd, weight, loss_collection='losses'):
    w_reg = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
    tf.add_to_collection(loss_collection, w_reg)


def get_activation(input, activation='linear'):
  if activation == 'linear':
    return input
  elif activation == 'relu':
    return tf.nn.relu(input)
  elif activation == 'sigmoid':
    return tf.nn.sigmoid(input)
  elif activation == 'tanh':
    return tf.nn.tanh(input)
  else:
    raise NotImplementedError('Get_Activation [%s] is not found' % activation)


def concat(tensors, axis):
  return tf.concat(tensors, axis)


def norm_layer(input, ntype='batch', *args, **kwargs):
  if ntype == 'instance':
    n_layer = tf.contrib.layers.instance_norm(input, *args, **kwargs)
  elif ntype == 'batch':
    n_layer = tf.contrib.layers.batch_norm(input, *args, **kwargs)
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % ntype)
  return n_layer


def conv2d(input_,
            output_dim,
            kernel_h=3,
            kernel_w=None,
            stride_h=1,
            stride_w=None,
            padding='SAME',
            initializer=None,
            use_bias = True,
            wd=None,
            reuse=None,
            name="conv2d"):
  """Get a 2d-Convolutional layer with non-linear mapping"""
  if kernel_w == None:
    kernel_w = kernel_h
  if stride_w == None:
    stride_w = stride_h
  if reuse == None:
    reuse = tf.AUTO_REUSE

  with tf.variable_scope(name, reuse = reuse):
    w = get_variable(name='w',
                  shape=[kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
                  initializer=initializer,
                  wd=wd)

    conv = tf.nn.conv2d(input_, w, strides=[1,stride_h, stride_w, 1], padding=padding)

    if use_bias:
      b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.nn.bias_add(conv, b)
    return conv


def conv_block(x,
                nf,
                k,
                s,
                p='SAME',
                use_bias=True,
                wd=None,
                ntype=None,
                reuse=None,
                name='conv_block'):
  """Get a 2d-convolutional block (conv-norm-relu)"""
  if reuse == None:
    reuse = tf.AUTO_REUSE

  with tf.variable_scope(name, reuse=reuse) as scope:
    x = conv2d(x, nf, kernel_h=k, stride_h=s, use_bias=use_bias, wd=wd, name='conv')
    if not ntype == None:
      x = norm_layer(x, ntype)
    x = tf.nn.relu(x)
  return x


def max_pool(x, k=2, s=2, p='SAME', name='pooling'):
  with tf.variable_scope(name) as scope:
    return tf.nn.max_pool(x, [1,s,s,1], [1,k,k,1], p, name=name)


def softmax(x, axis=None, name=None):
    return tf.nn.softmax(x, axis=axis, name=name)


def fc_layer(input_,
              output_dim,
              initializer = None,
              activation='linear',
              use_bias=True,
              wd=None,
              reuse=None,
              name='fc'):
  """Get a fully connected layer with nonlinear mapping"""
  if reuse == None:
    reuse = tf.AUTO_REUSE

  shape = input_.get_shape().as_list()
  with tf.variable_scope(name or "Linear", reuse=tf.AUTO_REUSE) as scope:
    if len(shape) > 2:
      input_ = tf.layers.flatten(input_)

    shape = input_.get_shape().as_list()
    w = get_variable(name='fc_w', shape=[shape[1], output_dim],
                              initializer=initializer, wd=wd)
    result = tf.matmul(input_, w)

    if use_bias:
      b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))
      result = tf.nn.bias_add(result, b)
    result = get_activation(result, activation)
    return result

def VQA_classifier(input, hidden_dim, output_dim, drop_p, training, name=None):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    hidden1 = fc_layer(input, hidden_dim, name='linear')
    hidden1 = tf.layers.batch_normalization(hidden1, training=training, renorm=True, name='bn')
    hidden1 = tf.nn.relu(hidden1)
    hidden1 = tf.layers.dropout(hidden1, rate=drop_p, training=training, name='hidden1_drop')
    hidden2 = fc_layer(hidden1, output_dim, name='logit')
    return hidden2

def compute_score_with_logits(logits, labels, output_dim, name='score'):
  with tf.name_scope(name) as scope:
    logits = tf.argmax(logits, axis=1)
    one_hots = tf.one_hot(logits, output_dim, name='pred_one_hot')
    scores = tf.multiply(labels, one_hots)
  return scores
