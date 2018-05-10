# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.util import nest

### Building Blocks ###

def fully_connected(inputs,
                    num_outputs,
                    activation_fn=None,
                    dropout=None,
                    is_training=True,
                    reuse=None,
                    scope=None):
    """Adds a fully connected layer.

    """

    if not isinstance(num_outputs, int):
        raise ValueError('num_outputs should be integer, got %s.', num_outputs)

    trainable = (is_training != False)
    with tf.variable_scope(scope,
                           'fully_connected',
                           [inputs],
                           reuse=reuse) as sc:
        dtype = inputs.dtype.base_dtype
        num_input_units = inputs.get_shape()[-1].value

        static_shape = inputs.get_shape().as_list()
        static_shape[-1] = num_outputs

        out_shape = tf.unstack(tf.shape(inputs))
        out_shape[-1] = num_outputs

        weights_shape = [num_input_units, num_outputs]
        weights = tf.contrib.framework.model_variable(
            'weights',
            shape=weights_shape,
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable)
        weights_norm = tf.contrib.framework.model_variable(
            'weights_norm',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=tf.GraphKeys.WEIGHTS if trainable else None,
            trainable=trainable)
        if trainable:
            norm_op = weights.assign(
                tf.nn.l2_normalize(
                    weights, 0) * tf.exp(weights_norm))
            norm_op = tf.cond(
                tf.logical_or(tf.cast(sc.reuse, tf.bool),tf.logical_not(tf.cast(is_training, tf.bool))),
                lambda: tf.zeros([]),
                lambda: norm_op)
            with tf.control_dependencies([norm_op]):
                weights = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda: tf.nn.l2_normalize(weights, 0)*tf.exp(weights_norm),
                    lambda: tf.identity(weights))
        biases = tf.contrib.framework.model_variable(
            'biases',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            collections=tf.GraphKeys.BIASES if trainable else None,
            trainable=trainable)

        if len(static_shape) > 2:
            # Reshape inputs
            inputs = tf.reshape(inputs, [-1, num_input_units])

        if dropout != None:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        outputs = tf.matmul(inputs, weights) + biases

        if activation_fn:
            outputs = activation_fn(outputs)

        if len(static_shape) > 2:
            # Reshape back outputs
            outputs = tf.reshape(outputs, tf.stack(out_shape))
            outputs.set_shape(static_shape)
        return outputs

# convolutional layer
def convolution2d(inputs,
                  output_sizes,
                  kernel_sizes,
                  dilation_rates=None,
                  pool_size=None,
                  group_size=None,
                  activation_fn=None,
                  dropout=None,
                  is_training=True,
                  reuse=None,
                  scope=None):
    """Adds a 2D convolution followed by a maxpool layer.

    """

    with tf.variable_scope(scope,
                           'Conv',
                           [inputs],
                           reuse=reuse) as sc:
        dtype = inputs.dtype.base_dtype
        num_filters_in = inputs.get_shape()[-1].value
        if type(output_sizes) == int:
            output_sizes = [output_sizes]
            kernel_sizes = [kernel_sizes]
        assert(len(output_sizes) == len(kernel_sizes))

        trainable = (is_training != False)
        if dropout != None:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)
        output_list = []
        for i in range(len(output_sizes)):
            with tf.variable_scope("conv"+str(i)):
                kernel_size = kernel_sizes[i]
                output_size = output_sizes[i]
                dilation_rate = None
                if dilation_rates != None:
                    dilation_rate = dilation_rates[i]
                weights_shape = list(kernel_size) + [num_filters_in, output_size]
                weights = tf.contrib.framework.model_variable(
                    name='weights',
                    shape=weights_shape,
                    dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=trainable)
                weights_norm = tf.contrib.framework.model_variable(
                    'weights_norm',
                    shape=[output_size,],
                    dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    collections=tf.GraphKeys.WEIGHTS if trainable else None,
                    trainable=trainable)
                if is_training != False:
                    norm_op = weights.assign(
                        tf.nn.l2_normalize(
                            weights, [0,1,2]) * tf.exp(weights_norm))
                    norm_op = tf.cond(
                        tf.logical_or(tf.cast(sc.reuse, tf.bool),tf.logical_not(tf.cast(is_training, tf.bool))),
                        lambda: tf.zeros([]),
                        lambda: norm_op)
                    with tf.control_dependencies([norm_op]):
                        weights = tf.cond(
                            tf.cast(is_training, tf.bool),
                            lambda: tf.nn.l2_normalize(weights, [0,1,2])*tf.exp(weights_norm),
                            lambda: tf.identity(weights))
                biases = tf.contrib.framework.model_variable(
                    name='biases',
                    shape=[output_size,],
                    dtype=dtype,
                    initializer=tf.zeros_initializer(),
                    collections=tf.GraphKeys.BIASES if trainable else None,
                    trainable=trainable)
                outputs = tf.nn.convolution(
                    inputs, weights, padding='SAME', dilation_rate=dilation_rate) + biases
                if group_size:
                    num_group = output_size // group_size
                    outputs = tf.stack(tf.split(outputs, num_group, axis=3), axis=-1)
                    outputs = tf.nn.l2_normalize(outputs, [1,2,3])
                    outputs = tf.concat(tf.unstack(outputs, axis=-1), axis=-1)
                output_list.append(outputs)

        if len(output_list) == 1:
            outputs = output_list[0]
        else:
            outputs = tf.concat(output_list, axis=-1)

        if pool_size:
            pool_shape = [1] + list(pool_size) + [1]
            outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')

        if activation_fn:
            outputs = activation_fn(outputs)

        return outputs


def mpconv2d(inputs,
             output_sizes,
             kernel_sizes,
             pool_size=None,
             group_size=None,
             activation_fn=None,
             dropout=None,
             is_training=True,
             reuse=None,
             scope=None):
    """Adds a 2D convolution followed by a maxpool layer.

    """

    with tf.variable_scope(scope,
                           'Conv',
                           [inputs],
                           reuse=reuse) as sc:
        dtype = inputs.dtype.base_dtype
        num_filters_in = inputs.get_shape()[-1].value
        if type(output_sizes) == int:
            output_sizes = [output_sizes]
            kernel_sizes = [kernel_sizes]
        assert(len(output_sizes) == len(kernel_sizes))
        if dropout != None:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)
        output_list = []
        for i in range(len(output_sizes)):
            with tf.variable_scope("conv"+str(i)):
                kernel_size = kernel_sizes[i]
                output_size = output_sizes[i]
                outputs = tf.nn.max_pool(inputs,
                                         [1]+kernel_size+[1],
                                         [1,1,1,1],
                                         'SAME')
                outputs = fully_connected(outputs,
                                          output_size,
                                          is_training=is_training,
                                          scope="fc")
                if group_size:
                    num_group = output_size // group_size
                    outputs = tf.stack(tf.split(outputs, num_group, axis=3), axis=-1)
                    outputs = tf.nn.l2_normalize(outputs, [1,2,3])
                    outputs = tf.concat(tf.unstack(outputs, axis=-1), axis=-1)
                output_list.append(outputs)

        if len(output_list) == 1:
            outputs = output_list[0]
        else:
            outputs = tf.concat(output_list, axis=-1)

        if pool_size:
            pool_shape = [1] + list(pool_size) + [1]
            outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')

        if activation_fn:
            outputs = activation_fn(outputs)

        return outputs


### Regularization ###

def params_decay(decay):
    """ Add ops to decay weights and biases

    """

    params = tf.get_collection_ref(
        tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)

    while len(params) > 0:
        p = params.pop()
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            p.assign(decay*p + (1-decay)*tf.truncated_normal(
                p.get_shape(), stddev=0.01)))


### Nets ###

def MLP(inputs,
        num_layers,
        hidden_size,
        output_size,
        activation_fn=tf.nn.relu,
        dropout=None,
        is_training=True,
        reuse=None,
        scope=None):
    """ a deep neural net with fully connected layers

    """

    with tf.variable_scope(scope,
                           "MLP",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if dropout == None:
            outputs = inputs
        else:
            outputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        # residual layers
        for i in range(num_layers-1):
            outputs = fully_connected(outputs,
                                      hidden_size,
                                      activation_fn=activation_fn,
                                      is_training=is_training,
                                      scope="layer"+str(i))
        outputs = fully_connected(outputs,
                                  output_size,
                                  is_training=is_training,
                                  scope="layer"+str(num_layers-1))
        return outputs

def ResDNN(inputs,
           num_layers,
           size,
           activation_fn=tf.nn.relu,
           dropout=None,
           is_training=True,
           reuse=None,
           scope=None):
    """ a deep neural net with fully connected layers

    """

    with tf.variable_scope(scope,
                           "ResDNN",
                           [inputs],
                           reuse=reuse) as sc:
        if dropout == None:
            inputs = inputs
        else:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        outputs = MLP(inputs,
                      2,
                      size,
                      size,
                      is_training=is_training,
                      scope="proj")
        # residual layers
        for i in range(num_layers):
            residual = MLP(outputs,
                           2,
                           size,
                           size,
                           is_training=is_training,
                           scope="mlp"+str(i))
            outputs += residual
        return outputs

def ResDNN2(inputs,
            num_layers,
            size,
            activation_fn=tf.nn.relu,
            dropout=None,
            is_training=True,
            reuse=None,
            scope=None):
    """ a deep neural net with fully connected layers

    """

    with tf.variable_scope(scope,
                           "ResDNN",
                           [inputs],
                           reuse=reuse) as sc:
        if dropout == None:
            inputs = inputs
        else:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        feats = fully_connected(inputs,
                                size,
                                is_training=is_training,
                                scope="proj")
        outputs = 0.0
        reuse = None
        # residual layers
        for i in range(num_layers):
            outputs += fully_connected(tf.nn.relu(feats),
                                       size,
                                       activation_fn=tf.nn.relu,
                                       is_training=is_training,
                                       reuse=reuse,
                                       scope="fw")
            feats -= fully_connected(outputs,
                                     size,
                                     activation_fn=tf.nn.relu,
                                     is_training=is_training,
                                     reuse=reuse,
                                     scope="bw")
            reuse=True
        outputs = fully_connected(outputs,
                                  size,
                                  is_training=is_training,
                                  scope="outputs")
        return outputs

def SIRDNN(inputs,
           num_layers,
           activation_fn=tf.nn.relu,
           dropout=None,
           is_training=True,
           reuse=None,
           scope=None):
    """ a deep neural net with fully connected layers

    """

    with tf.variable_scope(scope,
                           "ResDNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if dropout == None:
            outputs = inputs
        else:
            outputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        # residual layers
        ru = None
        for i in range(num_layers):
            outputs -= fully_connected(activation_fn(outputs),
                                       size,
                                       activation_fn=activation_fn,
                                       is_training=is_training,
                                       reuse=ru,
                                       scope=sc)
            ru = True
        return outputs

def SIRCNN(inputs,
           num_layers,
           kernel_sizes,
           pool_size,
           pool_layers=1,
           activation_fn=tf.nn.relu,
           dropout=None,
           is_training=True,
           reuse=None,
           scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    with tf.variable_scope(scope,
                           "ResCNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if not pool_size:
            pool_layers = 0
        if dropout == None:
            outputs = inputs
        else:
            outputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        # residual layers
        for j in range(pool_layers+1):
            if j > 0:
                pool_shape = [1] + list(pool_size) + [1]
                inputs = tf.nn.max_pool(outputs,
                                        pool_shape,
                                        pool_shape,
                                        padding='SAME')
                if dropout == None:
                    outputs = inputs
                else:
                    outputs = tf.cond(
                        tf.cast(is_training, tf.bool),
                        lambda: tf.nn.dropout(inputs, dropout),
                        lambda: inputs)
            with tf.variable_scope("layer{0}".format(j)) as sc:
                ru = None
                for i in range(num_layers):
                    inputs = outputs
                    for k, kernel_size in enumerate(kernel_sizes):
                        outputs -= convolution2d(activation_fn(inputs),
                                                 size,
                                                 kernel_size,
                                                 activation_fn=activation_fn,
                                                 is_training=is_training,
                                                 reuse=ru,
                                                 scope="k"+str(k))
                    ru = True
        return outputs

def ResCNN(inputs,
            num_layers,
            kernel_size,
            pool_size,
            pool_layers=1,
            activation_fn=tf.nn.relu,
            dropout=None,
            is_training=True,
            reuse=None,
            scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    with tf.variable_scope(scope,
                           "ResCNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if not pool_size:
            pool_layers = 0
        if dropout == None:
            outputs = inputs
        else:
            outputs = tf.nn.dropout(inputs, dropout)

        # residual layers
        for j in range(pool_layers+1):
            if j > 0:
                pool_shape = [1] + list(pool_size) + [1]
                inputs = tf.nn.max_pool(outputs,
                                        pool_shape,
                                        pool_shape,
                                        padding='SAME')
                if dropout == None:
                    outputs = inputs
                else:
                    outputs = tf.nn.dropout(inputs, dropout)
            with tf.variable_scope("layer{0}".format(j)) as sc:
                inputs = outputs
                for i in range(num_layers):
                    inputs = convolution2d(activation_fn(inputs),
                                             size,
                                             kernel_size,
                                             activation_fn=None,
                                             dropout=dropout,
                                             is_training=is_training,
                                             scope="cnn"+str(i))
                outputs = activation_fn(outputs+inputs)
        return outputs

def DGCNN(inputs,
          num_layers,
          dropout=None,
          is_training=True,
          reuse=None,
          scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    with tf.variable_scope(scope,
                           "DGCNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if dropout == None:
            inputs = inputs
        else:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)

        # residual layers
        dilate_size = 0
        for i in range(num_layers):
            inputs_proj = fully_connected(inputs,
                                          2*size,
                                          activation_fn=tf.nn.relu,
                                          is_training=is_training,
                                          scope="inputs_proj_"+str(i))
            pool_size = 1+2*dilate_size
            if pool_size > 1:
                inputs_proj = tf.nn.max_pool(inputs_proj,
                                             [1,1,pool_size,1],
                                             [1,1,1,1],
                                             'SAME')
            dilate_size = 2**i
            contexts = convolution2d(inputs_proj,
                                     [size],
                                     [[1,3]],
                                     dilation_rates=[[1,dilate_size]],
                                     is_training=is_training,
                                     scope="contexts_"+str(i))
            outputs_proj = MLP(tf.concat([inputs, contexts], axis=-1),
                               2,
                               2*size,
                               2*size,
                               is_training=is_training,
                               scope="outputs_proj_"+str(i))
            gates, convs = tf.split(outputs_proj, 2, axis=3)
            gates = tf.sigmoid(gates)
            outputs = (1.0-gates)*inputs + gates*convs
            inputs = outputs
        return outputs

### RNN ###
class GRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.tanh, linear=fully_connected):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._linear = linear

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = tf.split(fully_connected(tf.concat([inputs, state], 1),
                                             2 * self._num_units), 2, 1)
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = self._activation(self._linear(tf.concat([inputs, r * state], 1),
                                     self._num_units))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class RANCell(tf.contrib.rnn.RNNCell):
  """Recurrent Additive Unit cell."""

  def __init__(self, num_units, activation=tf.nn.relu, linear=fully_connected):
    self._num_units = num_units
    self._activation = activation
    self._linear = linear

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        i, f = tf.split(fully_connected(tf.concat([inputs, state], 1),
                                             2 * self._num_units), 2, 1)
        i, f = tf.sigmoid(i), tf.sigmoid(f)
      new_h = f * state + i * inputs
    return new_h, new_h


def attention(query,
              keys,
              values,
              masks,
              coverage=None,
              is_training=True):
    """ implements the attention mechanism

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """

    query = tf.expand_dims(query, 1)
    if coverage != None:
        c = tf.expand_dims(tf.expand_dims(coverage, 1), 3)
        size = query.get_shape()[-1].value
        feat = query + keys + tf.squeeze(
            convolution2d(c,
                          size,
                          [1, 5],
                          is_training=is_training,
                          scope="coverage"),
            axis=[1])
    else:
        feat = query + keys
    logits = fully_connected(
        tf.tanh(feat), 1, is_training=is_training, scope="attention")
    logits = tf.squeeze(logits, [2])
    probs = tf.where(masks, tf.nn.softmax(logits), tf.zeros(tf.shape(logits)))
    attn_dist = probs / tf.reduce_sum(probs, -1, keep_dims=True)
    attn_feat = tf.reduce_sum(tf.expand_dims(attn_dist, 2) * values, [1])
    if coverage != None:
        coverage += attn_dist
        return attn_feat, attn_dist, coverage
    else:
        return attn_feat, attn_dist, None

def attention2(query,
               keys,
               values,
               masks=None,
               coverage=None,
               is_training=True):
    """ implements the attention mechanism

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """

    single_query = False
    if len(query.get_shape()) == 2:
        single_query = True
        query = tf.expand_dims(query, 1)
    if coverage != None:
        coverage = None
    size = values.get_shape()[-1].value
    feat = tf.expand_dims(query, 2) + tf.expand_dims(keys, 1)
    gates = fully_connected(tf.tanh(feat),
                            size,
                            activation_fn=tf.sigmoid,
                            is_training=is_training,
                            scope="gates")
    attn_feats = tf.reduce_max(tf.expand_dims(values, 1) * gates, 2)
    if single_query:
        attn_feats = tf.squeeze(attn_feats, 1)
    return attn_feats

def dynamic_attention(query,
                      keys,
                      values,
                      masks,
                      coverage=None,
                      size=None,
                      is_training=True):
    """dynamic attend on lower layer inputs

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """

    if size == None:
        size = values.get_shape()[-1].value
    single_query = False
    if len(query.get_shape()) == 2:
        single_query = True
        query = tf.expand_dims(query, 1)
    attn_feat = tf.zeros(tf.concat([tf.shape(query)[:-1], [size,]], 0))
    num_querys = tf.shape(query)[1]
    query = tf.expand_dims(query, 2)
    keys = tf.expand_dims(keys, 1)
    values = tf.expand_dims(values, 1)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, num_querys, 1])
    attn_logits, votes = tf.split(
        fully_connected(tf.nn.relu(query + keys),
                        size+1,
                        is_training=is_training,
                        scope='votes'),
        [1, size],
        axis=-1)
    attn_logits = tf.nn.relu(tf.squeeze(attn_logits, axis=-1))
    attn_logits = tf.where(masks, attn_logits, tf.zeros(tf.shape(attn_logits)))
    for _ in range(3):
        attn_feat = tf.expand_dims(attn_feat, 3)
        logits = tf.squeeze(tf.matmul(votes, attn_feat), axis=-1)
        probs = tf.where(masks, tf.nn.softmax(logits), tf.zeros(tf.shape(logits)))
        probs = probs / tf.reduce_sum(probs, -1, keep_dims=True)
        attn_feat = tf.reduce_sum(tf.expand_dims(probs, 3) * votes, [2])
    if single_query:
        attn_feat = tf.squeeze(attn_feat, 1)
        attn_logits = tf.squeeze(attn_logits, 1)
        probs = tf.squeeze(probs, 1)
    if coverage != None:
        coverage += probs
        return attn_feat, attn_logits, coverage
    else:
        return attn_feat, attn_logits, None

def dynamic_route(num_outputs,
                  values,
                  masks,
                  size=None,
                  is_training=True):
    """dynamic attend on lower layer inputs

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """

    def squash(inputs):
        """squash the inputs into 0-1"""
        norm = tf.norm(inputs, axis=-1, keep_dims=True)
        outputs = norm / (tf.reduce_max(norm, axis=1, keep_dims=True)+1.0) * tf.nn.l2_normalize(inputs, -1)
        return outputs

    if size == None:
        size = values.get_shape()[-1].value
    seq_len = tf.expand_dims(tf.reduce_sum(tf.to_float(masks), 1, keep_dims=True), -1)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, num_outputs, 1])
    votes = MLP(tf.nn.relu(values),
                2,
                512,
                num_outputs*size,
                is_training=is_training,
                scope='votes')
    votes = tf.reshape(votes,
                       tf.concat([tf.shape(values)[:2], [num_outputs, size]], 0))
    votes = tf.transpose(votes, [0, 2, 1, 3])
    logits = tf.zeros(tf.stack([tf.shape(values)[0], num_outputs, tf.shape(values)[1]], 0))
    probs = tf.where(masks, tf.nn.softmax(logits, 1), tf.zeros(tf.shape(logits)))
    cap_num = tf.reduce_sum(probs, 2, keep_dims=True)
    cap_feat = tf.reduce_sum(tf.expand_dims(probs, 3)*votes, 2)

    def cond(cap_feat, cap_num, open_mask):
        return tf.reduce_any(open_mask)

    def body(cap_feat, cap_num, open_mask):
        cap_feat = tf.expand_dims(cap_feat, 3)
        logits = tf.squeeze(tf.matmul(votes, tf.nn.l2_normalize(cap_feat, 2)), 3)
        probs = tf.where(masks, tf.nn.softmax(logits, 1), tf.zeros(tf.shape(logits)))
        cap_num = tf.reduce_sum(probs, 2, keep_dims=True)
        new_cap_feat = tf.reduce_sum(tf.expand_dims(probs, 3)*votes, 2)
        cap_feat = tf.squeeze(cap_feat, 3)
        update = new_cap_feat - cap_feat
        open_mask = tf.greater(
            tf.reduce_max(
                tf.norm(update, axis=-1)/(tf.norm(cap_feat, axis=-1)+1e-12),
                axis=-1),
            1e-1)
        cap_feat = tf.where(open_mask, cap_feat+0.5*(new_cap_feat-cap_feat), cap_feat)
        return cap_feat, cap_num, open_mask
    cap_feat, cap_num, _ = tf.while_loop(cond,
                                   body,
                                   [cap_feat, cap_num, tf.ones([tf.shape(cap_feat)[0]], dtype=tf.bool)])
    return squash(cap_feat)

def dynamic_route2(num_outputs,
                   input_poses,
                   input_activations,
                   masks,
                   size=None,
                   is_training=True):
    """dynamic attend on lower layer inputs

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """
    with tf.variable_scope("dynamic_route") as sc:

        beta1 = tf.contrib.framework.model_variable(
            'beta1',
            shape=[num_outputs,],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=tf.GraphKeys.BIASES,
            trainable=True)

        beta2 = tf.contrib.framework.model_variable(
            'beta2',
            shape=[num_outputs,],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=tf.GraphKeys.BIASES,
            trainable=True)

        if size == None:
             size = input_poses.get_shape()[-1].value
        seq_len = tf.expand_dims(tf.reduce_sum(tf.to_float(masks), 1, keep_dims=True), -1)
        masks = tf.tile(tf.expand_dims(masks, 1), [1, num_outputs, 1])
        votes = MLP(input_poses,
                    2,
                    512,
                    num_outputs*size,
                    is_training=is_training,
                    scope='poses')
        votes = tf.reshape(votes,
                           tf.concat([tf.shape(input_poses)[:2], [num_outputs, size]], 0))
        votes = tf.transpose(votes, [0, 2, 1, 3])
        input_activations = tf.expand_dims(input_activations, 1)
        r = tf.ones(tf.stack([tf.shape(input_poses)[0], num_outputs, tf.shape(input_poses)[1]], 0))
        r /= tf.reduce_sum(r, axis=1, keep_dims=True)
        r = tf.where(masks, r, tf.zeros(tf.shape(r)))
        r *= input_activations
        r_sum = tf.reduce_sum(r, 2, keep_dims=True)
        output_poses = tf.reduce_sum(tf.expand_dims(r, 3)*votes, 2) / (r_sum+1e-12)
        output_vars = tf.reduce_sum(tf.expand_dims(r, 3)*tf.square(votes-tf.expand_dims(output_poses, 2)), 2) / (r_sum+1e-12)
        cost = tf.reduce_sum(tf.expand_dims(beta1, -1)+0.5*tf.log(output_vars+1e-12), axis=-1) * tf.squeeze(r_sum, 2)
        cost_mean = tf.reduce_mean(cost, axis=1, keep_dims=True)
        cost_stdv = tf.sqrt(tf.reduce_sum(tf.square(cost - cost_mean), axis=1, keep_dims=True) / num_outputs)
        output_activations = tf.sigmoid(beta2 + (cost_mean - cost) / (cost_stdv + 1e-12))
        output_activations = tf.sigmoid(tf.squeeze(r_sum, -1))
        sc.reuse_variables()

        def cond(output_poses, output_vars, output_activations, open_mask):
            return tf.reduce_any(open_mask)

        def body(output_poses, output_vars, output_activations, open_mask):
            probs = 1/tf.sqrt(tf.reduce_prod(2*math.pi*tf.expand_dims(output_vars, 2), axis=-1))*\
                tf.exp(tf.reduce_sum(-0.5*tf.square(votes-tf.expand_dims(output_poses, 2))/tf.expand_dims(output_vars, 2), axis=-1))
            r = probs * tf.expand_dims(output_activations, 2)
            r /= (tf.reduce_sum(r, axis=1, keep_dims=True)+1e-12)
            r = tf.where(masks, r, tf.zeros(tf.shape(r)))
            r *= input_activations
            r_sum = tf.reduce_sum(r, 2, keep_dims=True)
            new_output_poses = tf.reduce_sum(tf.expand_dims(r, 3)*votes, 2) / (r_sum+1e-12)
            new_output_vars = tf.reduce_sum(tf.expand_dims(r, 3)*tf.square(votes-tf.expand_dims(new_output_poses, 2)), 2) / (r_sum+1e-12)
            cost = (beta1 + tf.reduce_sum(0.5*tf.log(new_output_vars+1e-12), axis=-1)) * tf.squeeze(r_sum, 2)
            cost_mean = tf.reduce_mean(cost, axis=1, keep_dims=True)
            cost_stdv = tf.sqrt(tf.reduce_sum(tf.square(cost - cost_mean), axis=1, keep_dims=True) / num_outputs)
            new_output_activations = tf.sigmoid(beta2 + (cost_mean - cost) / (cost_stdv + 1e-12))
            update = new_output_poses - output_poses
            open_mask = tf.greater(
                tf.reduce_max(
                    tf.norm(update, axis=-1)/(tf.norm(output_poses, axis=-1)+1e-12),
                    axis=-1),
                1e-1)
            output_poses = tf.where(open_mask, new_output_poses, output_poses)
            output_vars = tf.where(open_mask, new_output_vars, output_vars)
            output_activations = tf.where(open_mask, new_output_activations, output_activations)
            return output_poses, output_vars, output_activations, open_mask

        #output_poses, output_vars, output_activations, _= tf.while_loop(
        #    cond,
        #    body,
        #    [output_poses, output_vars, output_activations, tf.ones([tf.shape(output_poses)[0]], dtype=tf.bool)])
    return output_poses, output_activations

def dynamic_route3(num_outputs,
                   input_poses,
                   input_activations,
                   masks,
                   size=None,
                   is_training=True):
    """dynamic attend on lower layer inputs

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """
    with tf.variable_scope("dynamic_route") as sc:

        beta = tf.contrib.framework.model_variable(
            'beta',
            shape=[num_outputs,],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            collections=tf.GraphKeys.BIASES,
            trainable=True)

        if size == None:
            size = input_poses.get_shape()[-1].value
        num_votes = 16
        batch_size = tf.shape(input_poses)[0]
        length = tf.shape(input_poses)[1]
        seq_len = tf.expand_dims(tf.reduce_sum(tf.to_float(masks), 1, keep_dims=True), -1)
        masks = tf.tile(tf.expand_dims(tf.expand_dims(masks, 1), 2), [1, num_outputs, num_votes, 1])
        #votes = MLP(input_poses,
        #            2,
        #            512,
        #            num_outputs*size,
        #            is_training=is_training,
        #            scope='votes')
        votes = fully_connected(input_poses,
                                num_votes*size,
                                is_training=is_training,
                                scope='votes')
        votes = tf.reshape(votes,
                           tf.concat([tf.shape(input_poses)[:2], [num_votes, size]], 0))
        votes = tf.transpose(votes, [0, 2, 1, 3])
        votes = tf.nn.relu(votes)
        votes = tf.tile(tf.expand_dims(votes, 1), [1,num_outputs,1,1,1])
        input_activations = tf.expand_dims(input_activations, 1)
        r = tf.ones(tf.stack([batch_size, num_outputs, num_votes, length], 0))
        r /= tf.reduce_sum(r, axis=[2], keep_dims=True)
        r = tf.where(masks, r, tf.zeros(tf.shape(r)))
        #r *= input_activations
        #r_sum = tf.reduce_sum(r, 2, keep_dims=True)
        #max_pool = tf.reduce_max(tf.expand_dims(r, 4)*votes, axis=[2,3])
        #output_poses = fully_connected(output_poses,
        #                               size,
        #                               is_training=is_training,
        #                               scope="output_poses")
        #expectations = tf.reduce_sum(tf.expand_dims(r, 3)*votes, axis=2) / r_sum
        max_pool = tf.zeros([batch_size, num_outputs, size])
        feat = fully_connected(max_pool,
                               num_outputs*(size+1),
                               is_training=is_training,
                               scope="feat")
        feat = tf.reshape(feat, [batch_size, num_outputs, num_outputs, size+1])
        idx0 = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_outputs])
        idx1 = tf.tile(tf.expand_dims(tf.range(num_outputs), 0), [batch_size, 1])
        idx = tf.stack([idx0, idx1, idx1], axis=2)
        feat = tf.gather_nd(feat, idx)
        output_poses, output_activations = tf.split(
            feat,
            [size, 1],
            axis=-1)
        output_activations = tf.squeeze(tf.sigmoid(output_activations), -1)

        def cond(output_poses, output_activations, open_mask, counter):
            return tf.logical_and(tf.reduce_any(open_mask), tf.less(counter, 5))

        def body(output_poses, output_activations, open_mask, counter):
            logits = tf.squeeze(tf.matmul(
                tf.reshape(votes, [batch_size,num_outputs,num_votes*length,size]),
                tf.expand_dims(output_poses, 3)), -1)
            r = tf.reshape(logits, [batch_size, num_outputs, num_votes, length])
            r = tf.nn.softmax(r, 2)
            #r *= tf.expand_dims(output_activations, 2)
            #r /= tf.reduce_sum(r, axis=[2], keep_dims=True)
            r = tf.where(masks, r, tf.zeros(tf.shape(r)))
           # r *= input_activations
           # r_sum = tf.reduce_sum(r, 2, keep_dims=True)
            #new_output_poses = fully_connected(new_output_poses,
            #                                   size,
            #                                   is_training=is_training,
            #                                   reuse=True,
            #                                   scope="output_poses")
            #expectations = tf.reduce_sum(tf.expand_dims(r, 3)*votes, axis=2) / r_sum
            max_pool = tf.reduce_max(tf.expand_dims(r, 4)*votes, axis=[2,3])
            feat = fully_connected(max_pool,
                                   num_outputs*(size+1),
                                   is_training=is_training,
                                   reuse=True,
                                   scope="feat")
            feat = tf.reshape(feat, [batch_size, num_outputs, num_outputs, size+1])
            feat = tf.gather_nd(feat, idx)
            new_output_poses, new_output_activations = tf.split(
                feat,
                [size, 1],
                axis=-1)
            new_output_activations = tf.squeeze(tf.sigmoid(new_output_activations), -1)
            update = new_output_poses - output_poses
            open_mask = tf.greater(
                tf.reduce_max(
                    tf.norm(update, axis=-1)/(tf.norm(output_poses, axis=-1)+1e-12),
                    axis=-1),
                1e-1)
            output_poses = tf.where(open_mask, new_output_poses, output_poses)
            output_activations = tf.where(open_mask, new_output_activations, output_activations)
            counter += 1
            return output_poses, output_activations, open_mask, counter

        output_poses, output_activations, _, _ = tf.while_loop(
            cond,
            body,
            [output_poses, output_activations, tf.ones([tf.shape(output_poses)[0]], dtype=tf.bool), 0])
    return output_poses, output_activations

def dynamic_route4(values,
                   masks,
                   size=None,
                   is_training=True):
    """dynamic attend on lower layer inputs

    values: [batch_size x length x num_modes x dim]
    """

    if size == None:
        size = values.get_shape()[-1].value
    batch_size = tf.shape(values)[0]
    length = tf.shape(values)[1]
    num_modes = tf.shape(values)[2]
    seq_len = tf.reduce_sum(tf.to_float(masks), 1, keep_dims=True)
    masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, num_modes])
    # votes [batch_size X length X num_modes X size]
    votes = fully_connected(tf.nn.relu(values),
                            size,
                            is_training=is_training,
                            scope='votes')
    # logits [batch_size X length X num_modes]
    logits = tf.zeros([batch_size, length, num_modes])
    probs = tf.where(masks, tf.nn.softmax(logits), tf.zeros(tf.shape(logits)))
    # cap_feat [batch_size X size]
    cap_feat = tf.reduce_sum(tf.expand_dims(probs, -1)*votes, [1,2]) / seq_len

    def cond(cap_feat, probs, open_mask):
        return tf.reduce_any(open_mask)

    def body(cap_feat, probs, open_mask):
        logits = tf.matmul(tf.reshape(votes, [batch_size, length*num_modes, size]),
                           tf.expand_dims(cap_feat, -1))
        logits = tf.reshape(logits, [batch_size, length, num_modes])
        probs = tf.where(masks, tf.nn.softmax(logits), tf.zeros(tf.shape(logits)))
        new_cap_feat = tf.reduce_sum(tf.expand_dims(probs, -1)*votes, [1,2]) / seq_len
        update = new_cap_feat - cap_feat
        open_mask = tf.greater(
            tf.norm(update, axis=1)/(tf.norm(cap_feat, axis=1)+1e-12),
            1e-1)
        cap_feat = tf.where(open_mask, cap_feat+0.5*update, cap_feat)
        return cap_feat, probs, open_mask
    _, probs, _ = tf.while_loop(cond,
                                body,
                                [cap_feat, probs, tf.ones([tf.shape(cap_feat)[0]], dtype=tf.bool)])
    values = tf.reduce_sum(values * tf.expand_dims(probs, -1), axis=2)
    return values

def dynamic_route5(values,
                   masks,
                   size=None,
                   is_training=True):
    """dynamic attend on lower layer inputs

    values: [batch_size x length x num_modes x dim]
    """

    if size == None:
        size = values.get_shape()[-1].value
    batch_size = tf.shape(values)[0]
    length = tf.shape(values)[1]
    num_modes = values.get_shape()[2].value

    seq_len = tf.reduce_sum(tf.to_float(masks), 1, keep_dims=True)
    maskss = tf.tile(tf.expand_dims(masks, -1), [1, 1, num_modes])
    # votes [batch_size X length X num_modes X size]
    votes = values
    # logits [batch_size X length X num_modes]

    vote_list = tf.unstack(votes, axis=-2)
    fancy_mask_list = []
    fancy_vote_list = []
    for i in range(len(vote_list)):
        rp = i//2
        lp = i - rp
        mask = tf.pad(masks, [[0,0], [lp,rp]])
        vote = tf.pad(vote_list[i], [[0,0], [lp,rp], [0,0]])
        for j in range(i+1):
            fancy_mask_list.append(tf.slice(mask, [0,j], [-1,length]))
            fancy_vote_list.append(tf.slice(vote, [0,j,0], [-1,length,-1]))
    fancy_masks = tf.stack(fancy_mask_list, axis=-1)
    fancy_votes = tf.stack(fancy_vote_list, axis=-2)

    # cap_feat [batch_size X size]
    logits = tf.zeros([batch_size, length, int((1+num_modes)*num_modes/2)])
    cap_feat = tf.reduce_sum(
        tf.expand_dims(tf.nn.softmax(logits), axis=-1) * fancy_votes,
        axis=2)

    def cond(cap_feat, open_mask):
        return tf.reduce_any(open_mask)

    def body(cap_feat, open_mask):
        logits = tf.matmul(fancy_votes,
                           tf.expand_dims(cap_feat, -1))
        logits = tf.reshape(logits, [batch_size, length, int((1+num_modes)*num_modes/2)])
        new_cap_feat = tf.reduce_sum(
            tf.expand_dims(tf.nn.softmax(logits), axis=-1) * fancy_votes,
            axis=2)
        update = new_cap_feat - cap_feat
        update = tf.reshape(update, [batch_size*length, size])
        cap_feat = tf.reshape(cap_feat, [batch_size*length, size])
        open_mask = tf.greater(
            tf.norm(update, axis=-1)/(tf.norm(cap_feat, axis=-1)+1e-12),
            1e-1)
        cap_feat = tf.where(open_mask, cap_feat+0.5*update, cap_feat)
        cap_feat = tf.reshape(cap_feat, [batch_size, length, size])
        return cap_feat, open_mask
    cap_feat, _ = tf.while_loop(cond,
                                body,
                                [cap_feat, tf.ones([tf.shape(cap_feat)[0]], dtype=tf.bool)])
    return cap_feat

class AttentionCellWrapper(tf.contrib.rnn.RNNCell):
    """Wrapper for attention mechanism"""

    def __init__(self,
                 cell,
                 num_attention=2,
                 self_attention_idx=0,
                 use_copy=None,
                 use_coverage=None,
                 attention_fn=attention,
                 is_training=True):
        self.cell = cell
        self.num_attention = num_attention
        self.self_attention_idx = self_attention_idx
        self.use_copy = use_copy if use_copy != None else [False]*num_attention
        assert(len(self.use_copy) == self.num_attention)
        self.use_coverage = use_coverage if use_coverage != None else [False]*num_attention
        assert(len(self.use_coverage) == self.num_attention)
        self.attention_fn = attention_fn
        self.is_training=is_training

    def __call__(self,
                 inputs,
                 state,
                 scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            keys = []
            values = []
            masks = []
            attn_feats = []
            coverages = []
            for i in range(self.num_attention):
                keys.append(state[0])
                values.append(state[1])
                masks.append(state[2])
                attn_feats.append(state[3])
                if self.use_coverage[i]:
                    coverages.append(state[4])
                    state = state[5:]
                else:
                    coverages.append(None)
                    state = state[4:]
            cell_state = state if len(state) > 1 else state[0]
            batch_size = tf.shape(inputs)[0]

            cell_outputs, cell_state = self.cell(inputs, cell_state)
            # update self attention
            if (self.self_attention_idx >= 0 and
                self.self_attention_idx < self.num_attention):
                value_size = values[self.self_attention_idx].get_shape()[-1].value
                key_size = keys[self.self_attention_idx].get_shape()[-1].value
                new_value = cell_outputs
                values[self.self_attention_idx] = tf.concat(
                    [values[self.self_attention_idx],
                     tf.reshape(new_value, [batch_size, 1, value_size])],
                    axis=1)
                new_key = fully_connected(new_value,
                                          key_size,
                                          is_training=self.is_training,
                                          scope="key")
                keys[self.self_attention_idx] = tf.concat(
                    [keys[self.self_attention_idx],
                     tf.reshape(new_key, [batch_size, 1, key_size])],
                    axis=1)
                new_mask = tf.ones([batch_size, 1], dtype=tf.bool)
                masks[self.self_attention_idx] = tf.concat(
                    [masks[self.self_attention_idx], new_mask],
                    axis=1)

            # attend
            key_sizes = []
            for i in range(self.num_attention):
                key_sizes.append(keys[i].get_shape()[-1].value)
            query_all = fully_connected(
                tf.concat([cell_outputs,] + attn_feats, axis=-1),
                sum(key_sizes),
                is_training=self.is_training,
                scope="query_proj")
            queries = tf.split(query_all, key_sizes, axis=-1)
            copy_logits = []
            for i in range(self.num_attention):
                query = queries[i]
                with tf.variable_scope("attention"+str(i)):
                    attn_feats[i], attn_logits, coverages[i] = self.attention_fn(
                        query,
                        keys[i],
                        values[i],
                        masks[i],
                        coverages[i],
                        is_training=self.is_training)
                if self.use_copy[i]:
                    copy_logits.append(attn_logits)
            outputs = [tf.concat([cell_outputs,] + attn_feats, 1),]
            outputs = tuple(
                outputs + [tuple(copy_logits),]
                if len(copy_logits) > 0 else outputs[0])
            cell_state = cell_state if isinstance(cell_state, (tuple, list)) else (cell_state,)
            state = []
            for i in range(self.num_attention):
                if self.use_coverage[i]:
                    state.extend([keys[i], values[i], masks[i], attn_feats[i], coverages[i]])
                else:
                    state.extend([keys[i], values[i], masks[i], attn_feats[i]])
            state = tuple(state) + cell_state
            return outputs, state

    def get_coverage_penalty(self, state):
        """ get the coverage from state """
        cov_penalties = []
        for i in range(self.num_attention):
            if self.use_coverage[i]:
                mask = tf.to_float(state[2])
                coverage = state[4]
                cov_penalty = tf.reduce_sum(
                    tf.log(tf.minimum(coverage+1e-5, 1.0)) * mask,
                    axis=-1) / tf.reduce_sum(mask, axis=-1)
                cov_penalties.append(cov_penalty)
                state = state[5:]
            else:
                state = state[4:]
        return cov_penalties


def create_cell(size,
                num_layers,
                cell_type="GRU",
                activation_fn=tf.tanh,
                linear=None,
                is_training=True):
    """create various type of rnn cells"""

    def _linear(inputs, num_outputs):
        """fully connected layers inside the rnn cell"""

        return fully_connected(
            inputs, num_outputs, is_training=is_training)

    if not linear:
        linear=_linear

    # build single cell
    if cell_type == "GRU":
        single_cell = GRUCell(size, activation=activation_fn, linear=linear)
    elif cell_type == "RAN":
        single_cell = RANCell(size, activation=activation_fn, linear=linear)
    elif cell_type == "LSTM":
        single_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True, cell_clip=5.0, num_proj=size)
    else:
        raise ValueError('Incorrect cell type! (GRU|LSTM)')
    cell = single_cell
    # stack multiple cells
    if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
    return cell


### Recurrent Decoders ###

def greedy_dec(length,
               initial_state,
               input_embedding,
               cell,
               logit_fn):
    """ A greedy decoder.

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))

    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    symbol = tf.argmax(logits, 1)
    seq = [symbol]
    mask = tf.not_equal(symbol, 0)
    tf.get_variable_scope().reuse_variables()
    for _ in range(length-1):

        inputs = tf.nn.embedding_lookup(input_embedding, symbol)

        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        symbol = tf.argmax(logits, 1)
        symbol = tf.where(mask, symbol, tf.zeros([batch_size], dtype=tf.int64))
        mask = tf.not_equal(symbol, 0)

        seq.append(symbol)

    return tf.expand_dims(tf.stack(seq, 1), 1)

def stochastic_dec(length,
                   initial_state,
                   input_embedding,
                   cell,
                   logit_fn,
                   num_candidates=1):
    """ A stochastic decoder.

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))

    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    symbol = tf.reshape(tf.multinomial(logits, num_candidates), [-1])
    mask = tf.equal(symbol, 0)
    seq = [symbol]
    tf.get_variable_scope().reuse_variables()

    beam_parents = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size), 1),
                [1, num_candidates]), [-1])
    if isinstance(state, tuple):
        state = tuple([tf.gather(s, beam_parents) for s in state])
    else:
        state = tf.gather(state, beam_parents)

    for _ in range(length-1):

        inputs = tf.nn.embedding_lookup(input_embedding, symbol)

        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
        symbol = tf.where(mask,
                          tf.to_int64(tf.zeros([batch_size*num_candidates])),
                          symbol)
        mask = tf.equal(symbol, 0)
        seq.append(symbol)

    return tf.reshape(tf.stack(seq, 1), [batch_size, num_candidates, length])

# beam decoder
def beam_dec(length,
             initial_state,
             input_embedding,
             cell,
             logit_fn,
             num_candidates=1,
             beam_size=100,
             gamma=0.65):
    """ A basic beam decoder

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))
    vocab_size = tf.shape(input_embedding)[0]

    # iter
    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    prev = tf.nn.log_softmax(logits)
    probs = tf.slice(prev, [0, 1], [-1, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(
        tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
    paths = tf.reshape(symbols, [-1, 1])

    candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
    scores = [tf.slice(prev, [0, 0], [-1, 1])]

    tf.get_variable_scope().reuse_variables()

    for i in range(length-1):

        if isinstance(state, tuple):
            state = tuple([tf.gather(s, beam_parent) for s in state])
        else:
            state = tf.gather(state, beam_parent)

        inputs = tf.reshape(tf.nn.embedding_lookup(input_embedding, symbols),
                            [-1, inputs_size])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(
            tf.nn.log_softmax(logits),
            [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(fn,
                                  paths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  back_prop=False,
                                  swap_memory=True,
                                  infer_shape=False)),
            [batch_size, beam_size])
        if gamma == 0.0:
            close_score = best_probs + tf.squeeze(
                tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        else:
            close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
                tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        candidates.append(tf.reshape(tf.pad(paths,
                                            [[0, 0], [0, length-1-i]],
                                            "CONSTANT"),
                                     [batch_size, beam_size, length]))
        scores.append(close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(
            tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(
            tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent,
            [-1])
        paths = tf.gather(paths, beam_parent)
        paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
    scores = tf.concat(scores, 1)
    best_scores, indices = tf.nn.top_k(scores, num_candidates)
    indices = tf.reshape(
        tf.expand_dims(
            tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices,
        [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices),
                                 [batch_size, num_candidates, length])

    return best_candidates, best_scores

# beam decoder
def stochastic_beam_dec(length,
                        initial_state,
                        input_embedding,
                        cell,
                        logit_fn,
                        num_candidates=1,
                        beam_size=100,
                        gamma=0.65,
                        cp=0.0):
    """ A stochastic beam decoder

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))
    vocab_size = tf.shape(input_embedding)[0]

    # iter
    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    prev = tf.nn.log_softmax(logits)
    probs = tf.slice(prev, [0, 1], [-1, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(
        tf.expand_dims(tf.range(batch_size), 1) + beam_parent,
        [-1])
    paths = tf.reshape(symbols, [-1, 1])

    mask = tf.expand_dims(
        tf.nn.in_top_k(prev, tf.zeros([batch_size], dtype=tf.int32),
                       beam_size),
        1)
    candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
    scores = [tf.slice(prev, [0, 0], [-1, 1])]

    tf.get_variable_scope().reuse_variables()

    for i in range(length-1):

        if isinstance(state, tuple):
            state = tuple([tf.gather(s, beam_parent) for s in state])
        else:
            state = tf.gather(state, beam_parent)

        inputs = tf.reshape(
            tf.nn.embedding_lookup(input_embedding, symbols),
            [-1, inputs_size])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(
            tf.nn.log_softmax(logits),
            [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        mask = tf.concat(
            [mask,
             tf.reshape(
                 tf.nn.in_top_k(tf.reshape(
                     prev, [-1, vocab_size]),
                     tf.zeros([batch_size*beam_size], dtype=tf.int32),
                     beam_size),
                 [batch_size, beam_size])],
            1)
        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(fn,
                                  paths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  swap_memory=True)),
            [batch_size, beam_size])
        close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
            tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        if cp > 0.0:
            cov_penalties = cell.get_coverage_penalty(state)
            cov_penalty = tf.reduce_sum(
                tf.stack(
                    cov_penalties,
                    axis=-1),
                axis=-1)
            close_score += tf.reshape(cp*cov_penalty, [batch_size, beam_size])
        candidates.append(tf.reshape(tf.pad(paths,
                                            [[0, 0],[0, length-1-i]],
                                            "CONSTANT"),
                                     [batch_size, beam_size, length]))
        scores.append(close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]),
                           [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent, [-1])
        paths = tf.gather(paths, beam_parent)
        paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
    scores = tf.concat(scores, 1)
    fillers = tf.tile(tf.expand_dims(tf.reduce_min(scores, 1) - 20.0, 1), [1, tf.shape(scores)[1]])
    scores = tf.where(mask, scores, fillers)
    indices = tf.to_int32(tf.multinomial(scores * (7**gamma), num_candidates))
    indices = tf.reshape(tf.expand_dims(tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices, [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices), [batch_size, num_candidates, length])
    best_scores = tf.reshape(tf.gather(tf.reshape(scores, [-1]), indices), [batch_size, num_candidates])

    return best_candidates, best_scores


### Copy Mechanism ###

def make_logit_fn(vocab_embedding,
                  copy_ids=None,
                  is_training=True):
    """implements logit function with copy mechanism

    """

    if copy_ids == None:
        def logit_fn(outputs):
            size = vocab_embedding.get_shape()[-1].value
            outputs_vocab = fully_connected(outputs,
                                           size+16,
                                           is_training=is_training,
                                           scope="proj")
            outputs_vocab_main, outputs_vocab_norm = tf.split(
                outputs_vocab,
                [size, 16],
                axis=-1)
            outputs_vocab_main = tf.stack(
                tf.split(outputs_vocab_main, 16, axis=-1), axis=-2)
            outputs_vocab_main = tf.nn.l2_normalize(outputs_vocab_main, -1)
            outputs_vocab = outputs_vocab_main * tf.nn.relu(
                tf.expand_dims(outputs_vocab_norm, axis=-1))
            outputs_vocab = tf.concat(
                tf.unstack(outputs_vocab, axis=-2),
                axis=-1)
            logits_vocab = tf.reshape(
                tf.matmul(tf.reshape(outputs_vocab,
                                     [-1, size]),
                          tf.transpose(vocab_embedding)),
                tf.concat([tf.shape(outputs)[:-1],
                           tf.constant(-1, shape=[1])], 0))
            return logits_vocab
    else:
        def logit_fn(outputs):
            batch_size = tf.shape(copy_ids[0])[0]
            size = vocab_embedding.get_shape()[-1].value
            vocab_size = vocab_embedding.get_shape()[0].value
            outputs, copy_logits = outputs
            assert(len(copy_ids) == len(copy_logits))
            outputs_vocab = fully_connected(outputs,
                                           size+16,
                                           is_training=is_training,
                                           scope="proj")
            outputs_vocab_main, outputs_vocab_norm = tf.split(
                outputs_vocab,
                [size, 16],
                axis=-1)
            outputs_vocab_main = tf.stack(
                tf.split(outputs_vocab_main, 16, axis=-1), axis=-2)
            outputs_vocab_main = tf.nn.l2_normalize(outputs_vocab_main, -1)
            outputs_vocab = outputs_vocab_main * tf.nn.relu(
                tf.expand_dims(outputs_vocab_norm, axis=-1))
            outputs_vocab = tf.concat(
                tf.unstack(outputs_vocab, axis=-2),
                axis=-1)
            if outputs.get_shape().ndims == 3:
                beam_size = outputs.get_shape()[1].value
                logits_vocab = tf.reshape(
                    tf.matmul(tf.reshape(outputs_vocab,
                                         [-1, size]),
                              tf.transpose(vocab_embedding)),
                    [batch_size, beam_size, vocab_size])
            else:
                assert(outputs.get_shape().ndims == 2)
                logits_vocab = tf.reshape(
                    tf.matmul(
                        outputs_vocab,
                        tf.transpose(vocab_embedding)),
                    [batch_size, -1, vocab_size])
            beam_size = tf.shape(logits_vocab)[1]
            logits = logits_vocab
            for i in range(len(copy_ids)):
                length = copy_ids[i].get_shape()[1].value
                data = tf.reshape(copy_logits[i], [batch_size, beam_size, length])
                batch_idx = tf.tile(
                    tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2),
                    [1, beam_size, length])
                beam_idx = tf.tile(
                    tf.expand_dims(tf.expand_dims(tf.range(beam_size), 0), 2),
                    [batch_size, 1, length])
                vocab_idx = tf.tile(
                    tf.expand_dims(copy_ids[i], 1),
                    [1, beam_size, 1])
                indices = tf.stack([batch_idx, beam_idx, vocab_idx], axis=-1)
                logits += tf.scatter_nd(indices, data, tf.shape(logits))
            if outputs.get_shape().ndims == 2:
                logits = tf.reshape(logits, [-1, vocab_size])
            return logits
    return logit_fn

