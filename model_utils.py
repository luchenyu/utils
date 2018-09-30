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
                    weight_normalize=True,
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
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    weights_collections = collections
    biases_collections = collections
    if trainable:
        weights_collections.append(tf.GraphKeys.WEIGHTS)
        biases_collections.append(tf.GraphKeys.BIASES)

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
        weights = tf.get_variable(
            'weights',
            shape=weights_shape,
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=trainable)
        weights_norm = tf.get_variable(
            'weights_norm',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=weights_collections,
            trainable=trainable)
        if trainable and weight_normalize:
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
        biases = tf.get_variable(
            'biases',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            collections=biases_collections,
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
                  weight_normalize=True,
                  activation_fn=None,
                  dropout=None,
                  is_training=True,
                  reuse=None,
                  scope=None):
    """Adds a 2D convolution followed by a maxpool layer.

    """

    trainable = (is_training != False)
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    weights_collections = collections
    biases_collections = collections
    if trainable:
        weights_collections.append(tf.GraphKeys.WEIGHTS)
        biases_collections.append(tf.GraphKeys.BIASES)

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
                dilation_rate = None
                if dilation_rates != None:
                    dilation_rate = dilation_rates[i]
                weights_shape = list(kernel_size) + [num_filters_in, output_size]
                weights = tf.get_variable(
                    name='weights',
                    shape=weights_shape,
                    dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    trainable=trainable)
                weights_norm = tf.get_variable(
                    'weights_norm',
                    shape=[output_size,],
                    dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    collections=weights_collections,
                    trainable=trainable)
                if is_training != False and weight_normalize:
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
                biases = tf.get_variable(
                    name='biases',
                    shape=[output_size,],
                    dtype=dtype,
                    initializer=tf.zeros_initializer(),
                    collections=biases_collections,
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
                tf.shape(p), stddev=0.01)))

### Optimize ###

def optimize_loss(loss,
                  global_step,
                  optimizer,
                  var_list=None,
                  scope=None):
    """ Optimize the model using the loss.

    """

    grad_var_list = optimizer.compute_gradients(loss, var_list)
    learning_rate = optimizer._lr

    candidates = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope) + \
                 tf.get_collection(tf.GraphKeys.BIASES, scope=scope)

    update_ops_ref = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    for op in update_ops:
        update_ops_ref.remove(op)
    update_ops = list(set(update_ops))
    for grad_var in grad_var_list:
        grad, var = grad_var
        if (grad == None) or (not var in candidates):
            continue
        update_ops.append(
            var.assign(
                (1.0 - 0.1*learning_rate)*var + \
                (0.1*learning_rate)*tf.truncated_normal(
                    tf.shape(var), stddev=0.01)))
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(
            grad_var_list,
            global_step=global_step)
    return train_op

### Nets ###

def GLU(inputs,
        output_size,
        dropout=None,
        is_training=True,
        reuse=None,
        scope=None):
    """ Gated Linear Units.

    """

    with tf.variable_scope(scope,
                           "GLU",
                           [inputs],
                           reuse=reuse) as sc:
        if dropout != None:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)
        projs = fully_connected(
            inputs,
            output_size,
            is_training=is_training,
            scope="projs")
        gates = fully_connected(
            inputs,
            output_size,
            activation_fn=tf.sigmoid,
            is_training=is_training,
            scope="gates")
        outputs = projs*gates
        return outputs

def GCU(inputs,
        output_size,
        kernel_size,
        dropout=None,
        is_training=True,
        reuse=None,
        scope=None):
    """ Gated Conv Units.

    """

    with tf.variable_scope(scope,
                           "GCU",
                           [inputs],
                           reuse=reuse) as sc:
        if dropout != None:
            inputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, dropout),
                lambda: inputs)
        projs = convolution2d(
            inputs,
            output_size,
            kernel_size,
            is_training=is_training,
            scope="projs")
        gates = convolution2d(
            inputs,
            output_size,
            kernel_size,
            activation_fn=tf.sigmoid,
            is_training=is_training,
            scope="gates")
        outputs = projs*gates
        return outputs

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

def highway(inputs,
            num_layers,
            activation_fn=tf.nn.relu,
            dropout=None,
            is_training=True,
            reuse=None,
            scope=None):
    """ a highway network

    """

    with tf.variable_scope(scope,
                           "highway",
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
            projs = fully_connected(outputs,
                                    2*size,
                                    is_training=is_training,
                                    scope="layer"+str(i))
            gates, feats = tf.split(projs, 2, axis=-1)
            gates = tf.sigmoid(gates)
            feats = activation_fn(feats)
            outputs = gates*feats + (1.0-gates)*outputs
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

def cudnn_lstm(num_layers,
               num_units,
               direction,
               input_shape,
               trainable=True):
    """Create a cudnn lstm."""

    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        direction=direction)
    lstm.build(input_shape)
    if trainable and tf.get_variable_scope().reuse != True:
        for var in lstm.trainable_variables:
            tf.add_to_collection(
                tf.GraphKeys.WEIGHTS,
                var)
    elif not trainable:
        train_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in lstm.trainable_variables:
            train_vars.remove(var)
    return lstm

def cudnn_lstm_legacy(inputs,
                      num_layers,
                      hidden_size,
                      direction,
                      is_training):
    """Create a cudnn lstm."""

    def cudnn_lstm_parameter_size(input_size, hidden_size):
        biases = 8 * hidden_size
        weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
        return biases + weights
    def direction_to_num_directions(direction):
        if direction == "unidirectional":
            return 1
        elif direction == "bidirectional":
            return 2
        else:
            raise ValueError("Unknown direction: %r." % (direction,))
    def estimate_cudnn_parameter_size(num_layers,
                                      input_size,
                                      hidden_size,
                                      input_mode,
                                      direction):
        num_directions = direction_to_num_directions(direction)
        params = 0
        isize = input_size
        for layer in range(num_layers):
            for direction in range(num_directions):
                params += cudnn_lstm_parameter_size(
                    isize, hidden_size
                )
            isize = hidden_size * num_directions
            return params

    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])

    cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers, hidden_size, input_size,
        input_mode="linear_input", direction=direction)

    est_size = estimate_cudnn_parameter_size(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode="linear_input",
        direction=direction)

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer(),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                     tf.GraphKeys.WEIGHTS])

    init_state = tf.tile(
        tf.zeros([2 * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])

    hiddens, output_h, output_c = cudnn_cell(
        inputs,
        input_h=init_state,
        input_c=init_state,
        params=cudnn_params,
        is_training=is_training)

    # Convert to batch major
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    output_h = tf.transpose(output_h, [1, 0, 2])
    output_c = tf.transpose(output_c, [1, 0, 2])

    return hiddens, tf.concat([output_h, output_c], axis=1)


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
    probs = tf.where(masks, tf.exp(logits), tf.zeros(tf.shape(logits)))
    attn_dist = probs / tf.reduce_sum(probs, -1, keep_dims=True)
    attn_feat = tf.reduce_sum(tf.expand_dims(attn_dist, 2) * values, [1])
    if coverage != None:
        coverage += attn_dist
        return attn_feat, attn_dist, coverage
    else:
        return attn_feat, attn_dist, None

def attention_simple(querys,
                     keys,
                     values,
                     num_head=1,
                     size=None,
                     masks=None,
                     dropout=None,
                     is_training=True,
                     reuse=None,
                     scope=None,):
    """ implements the attention mechanism

    querys: [batch_size x num_querys x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    masks: [batch_size x num_querys x length]
    """

    with tf.variable_scope(scope,
                           "attention",
                           [querys, keys, values],
                           reuse=reuse) as sc:
        single_query = False
        if len(querys.get_shape()) == 2:
            single_query = True
            querys = tf.expand_dims(querys, 1)
        if size == None:
            size = values.get_shape()[-1].value

        querys = fully_connected(
            querys,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="query_projs")
        querys = tf.stack(tf.split(querys, num_head, axis=-1), axis=1)

        keys = fully_connected(
            keys,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="key_projs")
        keys = tf.stack(tf.split(keys, num_head, axis=-1), axis=1)

        values = fully_connected(
            values,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="value_projs")
        values = tf.stack(tf.split(values, num_head, axis=-1), axis=1)
        

        logits = tf.matmul(querys, keys, transpose_b=True)
        logits /= tf.sqrt(tf.to_float(size))

        probs = tf.exp(logits) * tf.expand_dims(tf.to_float(masks), axis=1)
        probs /= (tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-20)
        attn_feats = tf.matmul(probs, values)
        attn_feats = tf.concat(tf.unstack(attn_feats, axis=1), axis=-1)
        if single_query:
            attn_feats = tf.squeeze(attn_feats, 1)
    return attn_feats

def attention_with_position(querys,
                            keys,
                            values,
                            size=None,
                            num_head=1,
                            masks=None,
                            query_position_idxs=None,
                            key_position_idxs=None,
                            use_position=True,
                            dropout=None,
                            is_training=True,
                            reuse=None,
                            scope=None,):
    """ implements the attention mechanism

    querys: [batch_size x num_querys x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    masks: [batch_size x num_querys x length]
    """

    with tf.variable_scope(scope,
                           "attention",
                           [querys, keys, values],
                           reuse=reuse) as sc:
        single_query = False
        if len(querys.get_shape()) == 2:
            single_query = True
            querys = tf.expand_dims(querys, 1)
            masks = tf.expand_dims(masks, 1) if masks != None else None
            if query_position_idxs != None:
                query_position_idxs = tf.expand_dims(query_position_idxs, 1)
        if size == None:
            size = values.get_shape()[-1].value

        batch_size = tf.shape(querys)[0]
        query_len = tf.shape(querys)[1]
        key_len = tf.shape(keys)[1]

        query_content_embeds = fully_connected(
            querys,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="query_content_projs")
        query_content_embeds = tf.stack(tf.split(query_content_embeds, num_head, axis=-1), axis=1)
        if use_position:
            if query_position_idxs == None:
                query_position_idxs = tf.tile(tf.expand_dims(tf.range(query_len), 0), [batch_size, 1])
            query_position_embeds = embed_position(query_position_idxs, size)
            query_position_embeds = tf.expand_dims(query_position_embeds, 1)
            query_position_deltas = tf.get_variable(
                'query_posit_deltas',
                shape=[num_head, size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=(is_training != False))
            query_position_deltas = tf.reshape(query_position_deltas, [1,num_head,1,size])
            query_position_embeds = translate_position_embeds(query_position_embeds, query_position_deltas)

        key_content_embeds = fully_connected(
            keys,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="key_content_projs")
        key_content_embeds = tf.stack(tf.split(key_content_embeds, num_head, axis=-1), axis=1)
        if use_position:
            if key_position_idxs == None:
                key_position_idxs = tf.tile(tf.expand_dims(tf.range(key_len), 0), [batch_size, 1])
            key_position_embeds = embed_position(key_position_idxs, size)
            key_position_embeds = tf.tile(tf.expand_dims(key_position_embeds, 1), [1,num_head,1,1])

        values = fully_connected(
            values,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="value_projs")
        values = tf.stack(tf.split(values, num_head, axis=-1), axis=1)

        logits = tf.matmul(query_content_embeds, key_content_embeds, transpose_b=True)
        if use_position:
            logits += tf.matmul(query_position_embeds, key_position_embeds, transpose_b=True)
        logits /= tf.sqrt(float(size))
        if masks != None:
            weights = tf.expand_dims(tf.to_float(masks), axis=1)
        else:
            weights = tf.ones(tf.shape(logits))
        logits -= tf.stop_gradient(
            tf.reduce_sum(logits*weights, axis=-1, keepdims=True) / \
            (tf.reduce_sum(weights, axis=-1, keepdims=True)+1e-20))
        logits = logits * weights

        probs = tf.nn.softmax(logits) * weights
        probs /= (tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-20)
        attn_feats = tf.matmul(probs, values)
       # attn_posits = tf.matmul(probs, values_position)
       # cos, sin = tf.split(query_posits, 2, axis=-1)
       # query_posits = tf.concat([cos, -sin], axis=-1)
       # query_posits = tf.tile(tf.expand_dims(query_posits, 1), [1,num_head,1,1])
       # attn_posits = translate_position_embeds(attn_posits, query_posits)
       # attn_feats += fully_connected(
       #     attn_posits,
       #     size/num_head,
       #     is_training=is_training,
       #     scope="attn_posit_projs")
        attn_feats = tf.concat(tf.unstack(attn_feats, axis=1), axis=-1)
        if single_query:
            attn_feats = tf.squeeze(attn_feats, 1)
    return attn_feats

def embed_position(position_idx,
                   size):
    """
    Get position_embeds of size [size] from position idx
    """

    to_squeeze = False
    if len(position_idx.get_shape()) == 1:
        to_squeeze = True
        position_idx = tf.expand_dims(position_idx, 1)
    batch_size = tf.shape(position_idx)[0]
    seq_len = tf.shape(position_idx)[1]

    position_idx = tf.to_float(position_idx)
    size = tf.to_float(size)
    position_j = 1. / tf.pow(10000., \
                             2 * tf.range(size / 2, dtype=tf.float32 \
                            ) / size)
    position_j = tf.tile(tf.expand_dims(tf.expand_dims(position_j, 0), 0), [batch_size,1,1])
    position_i = tf.expand_dims(position_idx, 2)
    position_ij = tf.matmul(position_i, position_j)
    position_embeds = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 2)

    if to_squeeze:
        position_embeds = tf.squeeze(position_embeds, [1])
    return position_embeds

def translate_position_embeds(position_embeds, delta):
    position_embeds = tf.stack(tf.split(position_embeds, 2, axis=-1), axis=-1)
    cos, sin = tf.split(delta, 2, axis=-1)
    #norm = tf.sqrt(tf.square(cos)+tf.square(sin)) + 1e-20
    #cos /= norm
    #sin /= norm
    new_cos = tf.reduce_sum(
        position_embeds*tf.stack([cos, -sin], axis=-1),
        axis=-1)
    new_sin = tf.reduce_sum(
        position_embeds*tf.stack([sin, cos], axis=-1),
        axis=-1)
    new_position_embeds = tf.concat([new_cos, new_sin], axis=-1)
    return new_position_embeds

def transformer(inputs,
                num_layers,
                masks=None,
                dropout=None,
                is_training=True,
                reuse=None,
                scope=None):
    """Transformer decoder
    """

    with tf.variable_scope(scope,
                           "Transformer",
                           [inputs],
                           reuse=reuse) as sc:
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        size = inputs.get_shape()[-1].value
        attn_masks = tf.sequence_mask(
            tf.tile(tf.expand_dims(tf.range(length), 0), [batch_size, 1]),
            maxlen=length)
        if masks != None:
            attn_masks = tf.logical_and(attn_masks, tf.expand_dims(masks, 2))
        position_idx = tf.tile(tf.expand_dims(tf.range(length), 0), [batch_size, 1])
        for i in range(num_layers):
            with tf.variable_scope("layer"+str(i)):
                pinputs = inputs + embed_position(position_idx, size)
                inputs += attention_simple(pinputs, pinputs, inputs,
                    num_head=8, size=size, masks=attn_masks, dropout=dropout, is_training=is_training)
                inputs = tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)
                inputs += MLP(
                    inputs,
                    2,
                    2*size,
                    size,
                    dropout=dropout,
                    is_training=is_training)
                inputs = tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)
        outputs = inputs
    return outputs

def transformer2(inputs,
                 encodes,
                 num_layers,
                 masks,
                 dropout=None,
                 is_training=True,
                 reuse=None,
                 scope=None):
    """Transformer decoder
    """

    with tf.variable_scope(scope,
                           "Transformer",
                           [inputs],
                           reuse=reuse) as sc:
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        size = inputs.get_shape()[-1].value
        masks = tf.logical_and(
            tf.logical_not(tf.eye(length, batch_shape=[batch_size], dtype=tf.bool)),
            masks)
        for i in range(num_layers):
            with tf.variable_scope("layer"+str(i)):
                inputs += attention_with_position(inputs, encodes, encodes,
                    num_head=4, size=size, masks=masks, dropout=dropout, is_training=is_training)
                inputs = tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)
                inputs += MLP(
                    inputs,
                    2,
                    size,
                    size,
                    dropout=dropout,
                    is_training=is_training)
                inputs = tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)
    outputs = inputs
    return outputs

def read(query,
         keys,
         values,
         num_group=1,
         masks=None,
         is_training=True):
    """ implements the attention mechanism

    query: [[batch_size] x query_size x dim]
    keys: [[batch_size] x length x dim]
    values: [[batch_size] x length x value_dim]
    """

    batch_mode = (len(query.get_shape()) == 3)
    if not batch_mode:
        query = tf.expand_dims(query, 0)
        keys = tf.expand_dims(keys, 0)
        values = tf.expand_dims(values, 0)
    batch_size = tf.shape(query)[0]
    query_size = tf.shape(query)[1]
    query_dim = query.get_shape()[-1].value
    key_size = tf.shape(keys)[1]
    key_dim = keys.get_shape()[-1].value
    assert(query_dim == key_dim)
    value_size = tf.shape(values)[1]
    value_dim = values.get_shape()[-1].value

    query = tf.reshape(
        query,
        [batch_size, query_size, num_group, query_dim / num_group])
    query = tf.transpose(query, [0, 2, 1, 3])
    query = tf.reshape(
        query,
        [batch_size*num_group, query_size, query_dim / num_group])
    keys = tf.reshape(
        keys,
        [batch_size, key_size, num_group, key_dim / num_group])
    keys = tf.transpose(keys, [0, 2, 1, 3])
    keys = tf.reshape(
        keys,
        [batch_size*num_group, key_size, key_dim / num_group])
    values = tf.reshape(
        values,
        [batch_size, value_size, num_group, value_dim / num_group])
    values = tf.transpose(values, [0, 2, 1, 3])
    values = tf.reshape(
        values,
        [batch_size*num_group, value_size, value_dim / num_group])
    logits = tf.matmul(
        query,
        keys,
        transpose_b=True)
    probs = tf.nn.softmax(logits)
    if masks != None:
        probs = tf.reshape(
            probs,
            [batch_size, num_group*query_size, key_size])
        masks = tf.tile(tf.expand_dims(masks, 1), [1, num_group*query_size, 1])
        probs = tf.where(masks, probs, tf.zeros(tf.shape(probs)))
        probs /= (tf.reduce_sum(probs, axis=-1, keepdims=True)+1e-12)
        probs = tf.reshape(
            probs,
            [batch_size*num_group, query_size, key_size])
    outputs = tf.matmul(
        probs,
        values)
    outputs = tf.reshape(
        outputs,
        [batch_size, num_group, query_size, value_dim / num_group])
    outputs = tf.transpose(outputs, [0, 2, 1, 3])
    outputs = tf.reshape(
        outputs,
        [batch_size, query_size, value_dim])
    if not batch_mode:
        outputs = tf.squeeze(outputs, 0)

    return outputs

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

class CudnnLSTMCell(object):
    """Wrapper of tf.contrib.cudnn_rnn.CudnnLSM"""

    def __init__(self,
                 num_layers,
                 num_units,
                 direction,
                 dropout=0.0,
                 is_training=True):
        self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=num_units,
            direction=direction,
            dropout=dropout)
        self.num_layers = num_layers
        self.num_dirs = 1 if direction=="unidirectional" else 2
        self.num_units = num_units
        self.trainable = is_training != False

    def __call__(self,
                 inputs,
                 state,
                 reuse=None,
                 scope=None):
        with tf.variable_scope(scope or "LSTM_Cell", reuse=reuse) as sc:

            batch_size = tf.shape(inputs)[0]            
            state = tuple(
                tf.transpose(
                    tf.reshape(
                        s, [batch_size, self.num_layers*self.num_dirs, self.num_units]), [1,0,2]) 
                for s in state)
            to_squeeze = False
            if inputs.shape.ndims == 2:
                to_squeeze = True
                inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.transpose(inputs, [1,0,2])
            outputs, state = self.cell(inputs, state, training=self.trainable)
            state = tuple(
                tf.reshape(
                    tf.transpose(
                        s, [1,0,2]), [batch_size, self.num_layers*self.num_dirs*self.num_units])
                for s in state)
            outputs = tf.transpose(outputs, [1,0,2])
            if to_squeeze:
                outputs = tf.squeeze(outputs, axis=[1])
        return outputs, state

class AttentionCell(object):
    """Attention is All you need!"""

    def __init__(self,
                 size,
                 num_layer=3,
                 dropout=None,
                 is_training=True):
        self.size = size
        self.num_layer = num_layer
        self.dropout = dropout
        self.is_training = is_training

    def __call__(self,
                 inputs,
                 state,
                 reuse=None,
                 scope=None):
        with tf.variable_scope(scope or "Attn_Cell", reuse=reuse) as sc:
            batch_size = tf.shape(inputs)[0]
            input_size = inputs.get_shape()[-1].value
            decoder_inputs = state[0]
            encodes = state[1::2]
            masks = state[2::2]
            to_squeeze = False
            if inputs.shape.ndims == 2:
                to_squeeze = True
                idx = decoder_inputs.size()
                decoder_inputs_tensor = tf.cond(
                    tf.greater(idx, 0),
                    lambda: tf.concat([tf.reshape(decoder_inputs.read(idx-1), [batch_size, -1, input_size]),
                        tf.expand_dims(inputs, 1)], axis=1),
                    lambda: tf.expand_dims(inputs, 1))
                state = tuple([decoder_inputs.write(idx, decoder_inputs_tensor)] + list(state[1:]))
                length = 1
            else:
                idx = decoder_inputs.size()
                decoder_inputs_tensor = tf.cond(
                    tf.greater(idx, 0),
                    lambda: tf.concat([tf.reshape(decoder_inputs.read(idx-1), [batch_size, -1, input_size]),
                        inputs], axis=1),
                    lambda: inputs)
                decoder_inputs_tensor = tf.reshape(decoder_inputs_tensor, [batch_size, -1, input_size])
                state = tuple([decoder_inputs.write(idx, decoder_inputs_tensor)] + list(state[1:]))
                length = tf.shape(inputs)[1]
            start_idx = tf.shape(decoder_inputs_tensor)[1] - length
            end_idx = tf.shape(decoder_inputs_tensor)[1]
            position_idxs = tf.tile(tf.expand_dims(tf.range(end_idx), 0), [batch_size, 1])
            dec_masks = tf.sequence_mask(position_idxs, maxlen=end_idx)
            masks = map(lambda m: tf.tile(tf.expand_dims(m, 1), [1,end_idx,1]) if m.shape.ndims == 2 else m, masks)
            masks = map(lambda m: tf.pad(m, [[0,0],[start_idx,0],[0,0]]), masks)
            inputs = fully_connected(
                decoder_inputs_tensor,
                self.size,
                is_training=self.is_training,
                scope="input_proj")
            for i in range(self.num_layer):
                with tf.variable_scope("layer_{:d}".format(i)):
                    inputs += attention_with_position(
                        inputs,
                        inputs,
                        inputs,
                        size=self.size,
                        num_head=4,
                        masks=dec_masks,
                        dropout=self.dropout,
                        is_training=self.is_training)
                    outputs = inputs
                    for i in range(len(encodes)):
                        outputs += attention_with_position(
                            inputs,
                            encodes[i],
                            encodes[i],
                            size=self.size,
                            num_head=4,
                            masks=masks[i],
                            dropout=self.dropout,
                            is_training=self.is_training)
                    inputs = tf.contrib.layers.layer_norm(outputs, begin_norm_axis=-1)
                    inputs += MLP(
                        inputs,
                        2,
                        2*self.size,
                        self.size,
                        dropout=self.dropout,
                        is_training=self.is_training)
                    inputs = tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)
            outputs = inputs[:,start_idx:]
            if to_squeeze:
                outputs = tf.squeeze(outputs, axis=[1])
        return outputs, state

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
                 reuse=None,
                 scope=None):
        with tf.variable_scope(scope or "Attn_Wrapper", reuse=reuse):
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
                step = values[self.self_attention_idx].size()
                query_position_idx = tf.reshape(step, [1])
                query_position_idx = tf.tile(query_position_idx, [batch_size])

            # attend
            query = tf.concat([cell_outputs,] + attn_feats, axis=-1)
            copy_logits = []
            for i in range(self.num_attention):
                k = keys[i]
                v = values[i]
                m = masks[i]
                if i == self.self_attention_idx:
                    sk = k.read(k.size()-1)
                    sk = tf.reshape(sk, [batch_size, -1, cell_outputs.get_shape()[-1].value])
                    sv = v.read(v.size()-1)
                    sv = tf.reshape(sv, [batch_size, -1, cell_outputs.get_shape()[-1].value])
                    sm = m.read(m.size()-1)
                    sm = tf.reshape(sm, [batch_size, -1])
                    k = sk
                    v = sv
                    m = sm
                with tf.variable_scope("attention"+str(i)):
                    attn_feats[i], attn_logits, coverages[i] = self.attention_fn(
                        query,
                        k,
                        v,
                        m,
                        coverages[i],
                        query_position_idx=query_position_idx,
                        is_training=self.is_training)
                if self.use_copy[i]:
                    copy_logits.append(attn_logits)
            outputs = [tf.concat([cell_outputs,] + attn_feats, 1),]
            outputs = tuple(
                outputs + [tuple(copy_logits),]) \
                if len(copy_logits) > 0 else outputs[0]
            cell_state = cell_state if isinstance(cell_state, (tuple, list)) else (cell_state,)
            if (self.self_attention_idx >= 0 and
                self.self_attention_idx < self.num_attention):
                i = self.self_attention_idx
                new_k = tf.concat([sk, tf.expand_dims(cell_outputs, 1)], axis=1)
                keys[i] = keys[i].write(keys[i].size(), new_k)
                new_v = tf.concat([sv, tf.expand_dims(cell_outputs, 1)], axis=1)
                values[i] = values[i].write(values[i].size(), new_v)
                new_m = tf.concat([sm, tf.ones([batch_size,1], dtype=tf.bool)], axis=1)
                masks[i] = masks[i].write(masks[i].size(), new_m)

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

### Glow ###
def glow(inputs,
         num_layers,
         num_iters,
         step_fn,
         is_training=True,
         reuse=None,
         scope=None):
    """ glow architecture """
    with tf.variable_scope(scope,
                           "Glow",
                           [inputs],
                           reuse=reuse) as sc:
        x = inputs
        zlist = []
        for i in range(num_iters):
            with tf.variable_scope("iter"+str(i)):
                xdim = x.get_shape()[-1].value
                for j in range(num_layers):
                    with tf.variable_scope("layer"+str(j)):
                        x = fully_connected(x,
                                            xdim,
                                            is_training=is_training,
                                            reuse=reuse,
                                            scope="permut")
                        xa,xb = tf.split(x, [xdim//2, xdim-xdim//2], axis=-1)
                        feat = step_fn(xb, xdim//2*2, is_training=is_training, reuse=reuse)
                        logs,t = tf.split(feat, 2, axis=-1)
                        s = tf.exp(logs)
                        ya = s*xa+t
                        yb = xb
                        y = tf.concat([ya, yb], axis=-1)
                        x = y
                if i < num_iters-1:
                    z,h = tf.split(y, [xdim//2, xdim-xdim//2], axis=-1)
                    zlist.append(z)
                    x = h
                else:
                    zlist.append(y)
        outputs = tf.concat(zlist, axis=-1)
    return outputs



### Recurrent Decoders ###

def gather_state(state, beam_parent):
    ta = tf.TensorArray(tf.int32, size=0)
    t = tf.zeros([], dtype=tf.int32)
    if type(state) == type(t):
        state = tf.gather(state, beam_parent)
    elif type(state) == type(ta):
        size = state.size()
        last = state.read(size-1)
        new = tf.gather(last, beam_parent)
        state = state.write(size, new)
    else:
        l = []
        for s in state:
            if type(s) == type(t):
                l.append(tf.gather(s, beam_parent))
            else:
                size = s.size()
                last = s.read(size-1)
                new = tf.gather(last, beam_parent)
                s = s.write(size, new)
                l.append(s)
        state = tuple(l)
    return state

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

    batch_size = tf.shape(initial_state[-1])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value

    state = initial_state
    beam_parent = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1),
        [1, num_candidates])
    beam_parent = tf.reshape(beam_parent, [batch_size*num_candidates])
    state = gather_state(state, beam_parent)
    mask = tf.zeros([batch_size*num_candidates], dtype=tf.bool)

    seqs = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    seqs = seqs.write(0, tf.zeros([batch_size*num_candidates], dtype=tf.int32))
    scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    scores = scores.write(0, tf.zeros([batch_size*num_candidates]))

    def cond(seqs, scores, state, mask, i):
        return tf.less(i, length)

    def body(seqs, scores, state, mask, i):
        inputs = tf.nn.embedding_lookup(input_embedding, seqs.read(i))

        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
        score = tf.reduce_sum(tf.one_hot(symbol, tf.shape(logits)[1]) * logits, axis=1)
        symbol = tf.where(mask,
                          tf.zeros([batch_size*num_candidates], dtype=tf.int32),
                          tf.to_int32(symbol))
        score *= tf.to_float(tf.logical_not(mask))
        mask = tf.equal(symbol, 0)
        seqs = seqs.write(i+1, symbol)
        scores = scores.write(i+1, score)
        i += 1
        return seqs, scores, state, mask, i

    seqs, scores, state, mask, i = tf.while_loop(
        cond, body, [seqs, scores, state, mask, 0],
        back_prop=False, swap_memory=True)
    candidates = tf.reshape(tf.transpose(seqs.stack(), [1,0])[:,1:], [batch_size, num_candidates, length])
    scores = tf.reshape(
        tf.reduce_sum(scores.stack(), axis=0),
        [batch_size, num_candidates])

    return candidates, scores

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

    batch_size = tf.shape(initial_state[-1])[0] \
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
    beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size), 1)+beam_parent, [-1])
    paths = tf.reshape(symbols, [-1, 1])

    state = gather_state(state, beam_parent)

    tf.get_variable_scope().reuse_variables()
    paths = tf.TensorArray(tf.int32, size=0,
        dynamic_size=True, infer_shape=False).write(0, paths)
    candidates = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

    def cond(paths, candidates, scores, state, best_probs, i):
        return tf.less(i, length)

    def body(paths, candidates, scores, state, best_probs, i):

        pths = paths.read(i)
        pths = tf.reshape(pths, [tf.shape(best_probs)[0]*tf.shape(best_probs)[1], -1])
        inputs = tf.nn.embedding_lookup(input_embedding, pths[:,-1])

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
                                  pths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  back_prop=False,
                                  swap_memory=True)),
            [batch_size, beam_size])
        close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
            tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])

        candidates = candidates.write(i, tf.reshape(
            tf.pad(pths, [[0, 0],[0, length-i-1]], "CONSTANT"),
            [batch_size, beam_size, length]))
        scores = scores.write(i, close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size)*beam_size, 1)+beam_parent, [-1])
        pths = tf.gather(pths, beam_parent)
        pths = tf.concat([pths, tf.reshape(symbols, [-1, 1])], axis=1)
        paths = paths.write(i+1, pths)

        state = gather_state(state, beam_parent)
        i += 1

        return paths, candidates, scores, state, best_probs, i

    _, candidates, scores, _, _, _ = tf.while_loop(
        cond, body, [paths, candidates, scores, state, best_probs, 0],
        back_prop=False, parallel_iterations=128, swap_memory=True)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.transpose(candidates.stack(), [1,0,2,3]), [-1, length])
    scores = tf.reshape(tf.transpose(scores.stack(), [1,0,2]), [batch_size, -1])
    best_scores, indices = tf.nn.top_k(scores, num_candidates)
    indices = tf.reshape(
        tf.expand_dims(
            tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices,
        [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices), [batch_size, num_candidates, length])

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

    batch_size = tf.shape(initial_state[-1])[0] \
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
    beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size), 1)+beam_parent, [-1])
    paths = tf.reshape(symbols, [-1, 1])

    state = gather_state(state, beam_parent)

    tf.get_variable_scope().reuse_variables()
    paths = tf.TensorArray(tf.int32, size=0,
        dynamic_size=True, infer_shape=False).write(0, paths)
    masks = tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False)
    candidates = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

    def cond(paths, masks, candidates, scores, state, best_probs, i):
        return tf.less(i, length)

    def body(paths, masks, candidates, scores, state, best_probs, i):

        pths = paths.read(i)
        pths = tf.reshape(pths, [tf.shape(best_probs)[0]*tf.shape(best_probs)[1], -1])
        inputs = tf.nn.embedding_lookup(input_embedding, pths[:,-1])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(
            tf.nn.log_softmax(logits),
            [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        masks = masks.write(
            i, tf.reshape(
                tf.nn.in_top_k(tf.reshape(
                    prev, [-1, vocab_size]),
                    tf.zeros([batch_size*beam_size], dtype=tf.int32),
                    beam_size),
                [batch_size, beam_size]))

        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(fn,
                                  pths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  back_prop=False,
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

        candidates = candidates.write(i, tf.reshape(
            tf.pad(pths, [[0, 0],[0, length-i-1]], "CONSTANT"),
            [batch_size, beam_size, length]))
        scores = scores.write(i, close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size)*beam_size, 1)+beam_parent, [-1])
        pths = tf.gather(pths, beam_parent)
        pths = tf.concat([pths, tf.reshape(symbols, [-1,1])], axis=1)
        paths = paths.write(i+1, pths)

        state = gather_state(state, beam_parent)
        i += 1

        return paths, masks, candidates, scores, state, best_probs, i

    _, masks, candidates, scores, _, _, _ = tf.while_loop(
        cond, body, [paths, masks, candidates, scores, state, best_probs, 0],
        back_prop=False, parallel_iterations=128, swap_memory=True)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.transpose(candidates.stack(), [1,0,2,3]), [-1, length])
    scores = tf.reshape(tf.transpose(scores.stack(), [1,0,2]), [batch_size, -1])
    masks = tf.reshape(tf.transpose(masks.stack(), [1,0,2]), [batch_size, -1])
    fillers = tf.tile(tf.expand_dims(tf.reduce_min(scores, 1) - 20.0, 1), [1, tf.shape(scores)[1]])
    scores = tf.where(masks, scores, fillers)
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


### Miscellaneous ###

def slice_fragments(inputs, starts, lengths):
    """ Extract the documents_features corresponding to choosen sentences.
    Since sentences are different lengths, this will be jagged. Therefore,
    we extract the maximum length sentence and then pad appropriately.

    Arguments:
        inputs: [batch, time, features]
        starts: [batch, beam_size] starting locations
        lengths: [batch, beam_size] how much to trim.

    Returns:
        fragments: [batch, beam_size, max_length, features]
    """
    batch = tf.shape(inputs)[0]
    time = tf.shape(inputs)[1]
    beam_size = tf.shape(starts)[1]
    features = inputs.get_shape()[-1].value

    # Collapse the batch and time dimensions
    inputs = tf.reshape(
        inputs, [batch * time, features])

    # Compute the starting location of each sentence and adjust
    # the start locations to account for collapsed time dimension.
    starts += tf.expand_dims(time * tf.range(batch), 1)
    starts = tf.reshape(starts, [-1])

    # Gather idxs are consecutive rows beginning at start
    # and ending at start + length, for each start in starts.
    # If starts is [0; 6] and length is [0, 1, 2], then the
    # result is [0, 1, 2; 6, 7, 8], which is flattened to
    # [0; 1; 2; 6; 7; 8].
    # Ensure length is at least 1.
    max_length = tf.maximum(tf.reduce_max(lengths), 1)
    gather_idxs = tf.reshape(tf.expand_dims(starts, 1) +
                             tf.expand_dims(tf.range(max_length), 0), [-1])

    # Don't gather out of bounds
    gather_idxs = tf.minimum(gather_idxs, tf.shape(inputs)[0] - 1)

    # Pull out the relevant rows and partially reshape back.
    fragments = tf.gather(inputs, gather_idxs)
    fragments = tf.reshape(fragments, [batch * beam_size, max_length, features])

    # Mask out invalid entries
    length_mask = tf.sequence_mask(tf.reshape(lengths, [-1]), max_length)
    length_mask = tf.expand_dims(tf.cast(length_mask, tf.float32), 2)

    fragments *= length_mask

    return tf.reshape(fragments, [batch, beam_size, max_length, features])

def slice_words(seqs, segs, get_idxs=False, encodes=None):
    """
    slice seqs into pieces of words.
    
    seqs: seqs to slice
    segs: segmentation labels indicate where to cut
    """

    max_length = tf.shape(seqs)[1]
    padded_seqs = tf.pad(seqs, [[0,0],[1,1]])
    padded_segs = tf.pad(segs, [[0,0],[1,1]], constant_values=1.0)
    padded_segs = tf.cast(padded_segs, tf.bool)
    padded_seq_masks = tf.greater(padded_seqs, 0)
    padded_seg_masks1 = tf.logical_or(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_and(padded_segs, padded_seg_masks1)
    padded_seg_masks2 = tf.logical_xor(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_or(padded_segs, padded_seg_masks2)

    num_words = tf.reduce_sum(tf.to_int32(padded_segs), axis=1)-1
    max_num_word = tf.maximum(tf.reduce_max(num_words), 1)

    def get_idx(inputs):
        padded_seg, num_word = inputs
        idx = tf.range(max_length+1, dtype=tf.int32)
        idx = tf.boolean_mask(idx, padded_seg)
        start = tf.pad(idx[:-1], [[0,max_num_word-num_word]])
        start = tf.reshape(start, [max_num_word])
        length = tf.pad(idx[1:] - idx[:-1], [[0,max_num_word-num_word]])
        length = tf.reshape(length, [max_num_word])
        return start, length

    starts, lengths = tf.map_fn(
        get_idx,
        [padded_segs, num_words],
        (tf.int32, tf.int32),
        parallel_iterations=128,
        back_prop=False,
        swap_memory=True)

    results = []

    segmented_seqs = slice_fragments(
        tf.to_float(tf.expand_dims(seqs, axis=-1)),
        starts,
        lengths)
    segmented_seqs = tf.to_int32(tf.squeeze(segmented_seqs, axis=-1))
    results.append(segmented_seqs)

    if get_idxs:
        idx_starts = tf.to_int32(tf.logical_not(tf.sequence_mask(starts, max_length)))
        idx_ends = tf.sequence_mask(starts+lengths, max_length, dtype=tf.int32)
        idx_ends *= tf.expand_dims(tf.expand_dims(tf.range(1, max_num_word+1, dtype=tf.int32), 0), 2)
        segment_idxs = tf.reduce_sum(idx_starts * idx_ends, axis=1)-1
        results.append(segment_idxs)

    if encodes != None:
        segmented_encodes = slice_fragments(
            encodes,
            starts,
            lengths)
        results.append(segmented_encodes)

    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)

def stitch_chars(segmented_seqs):
    """
    stitch segmented seqs into seq of chars.
    
    segmented_seqs: seqs to stitch
    """

    masks = tf.greater(segmented_seqs, 0)
    num_chars = tf.reduce_sum(tf.to_int32(masks), axis=[1,2])
    max_num_char = tf.reduce_max(num_chars)

    def stitch(segmented_seq):
        mask = tf.greater(segmented_seq, 0)
        seq = tf.boolean_mask(
            segmented_seq,
            mask)
        num_char = tf.shape(seq)[0]
        seq = tf.pad(seq, [[0,max_num_char-num_char]])
        return seq

    seqs = tf.map_fn(
        stitch,
        segmented_seqs,
        tf.int32,
        parallel_iterations=128,
        back_prop=False,
        swap_memory=True)
    return seqs
