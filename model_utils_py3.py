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
from collections import namedtuple, Iterable
import tensorflow as tf
from tensorflow.python.util import nest

### Building Blocks ###

def fully_connected(inputs,
                    num_outputs,
                    weight_normalize=False,
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
            initializer=tf.initializers.variance_scaling(mode='fan_in'),
            trainable=trainable,
            aggregation=tf.VariableAggregation.MEAN)
        if trainable and weight_normalize:
            weights_norm = tf.get_variable(
                'weights_norm',
                shape=[num_outputs,],
                dtype=dtype,
                initializer=tf.initializers.variance_scaling(mode='fan_out', distribution='uniform'),
                collections=weights_collections,
                trainable=trainable,
                aggregation=tf.VariableAggregation.MEAN)
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
                    lambda: tf.nn.l2_normalize(weights, 0) * tf.exp(weights_norm),
                    lambda: tf.identity(weights))
        biases = tf.get_variable(
            'biases',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.initializers.zeros(),
            collections=biases_collections,
            trainable=trainable,
            aggregation=tf.VariableAggregation.MEAN)

        if len(static_shape) > 2:
            # Reshape inputs
            inputs = tf.reshape(inputs, [-1, num_input_units])

        if dropout != None:
            if is_training == True:
                inputs = tf.nn.dropout(inputs, rate=dropout)
            elif is_training != False:
                inputs = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda: tf.nn.dropout(inputs, rate=dropout),
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
                  weight_normalize=False,
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
            if is_training == True:
                inputs = tf.nn.dropout(inputs, rate=dropout)
            elif is_training != False:
                inputs = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda: tf.nn.dropout(inputs, rate=dropout),
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
                    shape=[kernel_size[0]*kernel_size[1]*num_filters_in, output_size],
                    dtype=dtype,
                    initializer=tf.initializers.variance_scaling(mode='fan_in'),
                    trainable=trainable,
                    aggregation=tf.VariableAggregation.MEAN)
                if is_training != False and weight_normalize:
                    weights_norm = tf.get_variable(
                        'weights_norm',
                        shape=[output_size,],
                        dtype=dtype,
                        initializer=tf.initializers.variance_scaling(mode='fan_out', distribution='uniform'),
                        collections=weights_collections,
                        trainable=trainable,
                        aggregation=tf.VariableAggregation.MEAN)
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
                            lambda: tf.nn.l2_normalize(weights, 0) * tf.exp(weights_norm),
                            lambda: tf.identity(weights))
                biases = tf.get_variable(
                    name='biases',
                    shape=[output_size,],
                    dtype=dtype,
                    initializer=tf.initializers.zeros(),
                    collections=biases_collections,
                    trainable=trainable,
                    aggregation=tf.VariableAggregation.MEAN)
                weights = tf.reshape(weights, weights_shape)
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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

def layer_norm(inputs,
               begin_norm_axis=-1,
               is_training=True,
               scope=None):
    """Simple wrapper of tf.contrib.layers.layer_norm.

    """

    with tf.variable_scope(scope, 'layer_norm') as sc:

        dtype = inputs.dtype.base_dtype
        trainable = (is_training != False)
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if trainable:
            collections.append(tf.GraphKeys.WEIGHTS)
        beta = tf.get_variable(
            'beta',
            shape=[],
            dtype=dtype,
            initializer=tf.initializers.zeros(),
            collections=collections,
            trainable=trainable,
            aggregation=tf.VariableAggregation.MEAN)
        gamma = tf.get_variable(
            'gamma',
            shape=[],
            dtype=dtype,
            initializer=tf.initializers.ones(),
            collections=collections,
            trainable=trainable,
            aggregation=tf.VariableAggregation.MEAN)
        outputs = tf.contrib.layers.layer_norm(
            inputs, center=False, scale=False, begin_norm_axis=begin_norm_axis)
        outputs = beta + gamma*outputs
    return outputs


### Optimize ###

def optimize_loss(loss,
                  global_step,
                  optimizer,
                  wd=.0,
                  var_list=None,
                  scope=None):
    """ Optimize the model using the loss.

    """

    wd = .0 if wd == None else wd
    var_list = tf.trainable_variables(scope=scope) if var_list == None else var_list
    grad_var_list = optimizer.compute_gradients(
        loss, var_list, aggregation_method=tf.AggregationMethod.DEFAULT)

    candidates = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope) + \
                 tf.get_collection(tf.GraphKeys.BIASES, scope=scope)

    update_ops_ref = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    for op in update_ops:
        update_ops_ref.remove(op)
    update_ops = list(set(update_ops))
    if wd > .0:
        wd_every = 10
        wd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=wd_every*wd)
        wd_grad_var_list = []
        for i, grad_var in enumerate(grad_var_list):
            grad, var = grad_var
            if (grad == None) or (not var in candidates):
                continue
            wd_grad_var_list.append((var, var))
        wd_op = wd_optimizer.apply_gradients(
            wd_grad_var_list)
        wd_op = tf.cond(
            tf.equal(tf.floormod(global_step, wd_every), 0),
            lambda: wd_op,
            lambda: tf.no_op())
        update_ops.append(wd_op)
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
        feats = fully_connected(
            inputs,
            2*output_size,
            dropout=dropout,
            is_training=is_training,
            scope="feats")
        projs, gates = tf.split(feats, 2, axis=-1)
        gates = tf.sigmoid(gates)
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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
        if dropout != None:
            if is_training == True:
                inputs = tf.nn.dropout(inputs, rate=dropout)
            elif is_training != False:
                inputs = tf.cond(
                    tf.cast(is_training, tf.bool),
                    lambda: tf.nn.dropout(inputs, rate=dropout),
                    lambda: inputs)
        outputs = inputs

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
                lambda: tf.nn.dropout(inputs, rate=dropout),
                lambda: inputs)

        # residual layers
        for i in range(num_layers-1):
            projs = fully_connected(activation_fn(outputs),
                                    2*size,
                                    is_training=is_training,
                                    scope="layer"+str(i))
            gates, feats = tf.split(projs, 2, axis=-1)
            gates = tf.sigmoid(gates)
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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
                        lambda: tf.nn.dropout(inputs, rate=dropout),
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
            outputs = tf.nn.dropout(inputs, rate=dropout)

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
                    outputs = tf.nn.dropout(inputs, rate=dropout)
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
                lambda: tf.nn.dropout(inputs, rate=dropout),
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


### Attention ###

def attention_simple(querys,
                     keys,
                     values,
                     num_heads=1,
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

        querys = tf.stack(tf.split(querys, num_heads, axis=-1), axis=1)

        keys = tf.stack(tf.split(keys, num_heads, axis=-1), axis=1)

        values = tf.stack(tf.split(values, num_heads, axis=-1), axis=1)
        
        logits = tf.matmul(querys, keys, transpose_b=True)

        if masks != None:
            weights = tf.expand_dims(tf.cast(masks, tf.float32), axis=1)
        else:
            weights = tf.ones(tf.shape(logits))
        logits *= weights * tf.sqrt(1.0/tf.cast(size, tf.float32))
        logits = tf.pad(logits, [[0,0], [0,0], [0,0], [1,0]])
        weights = tf.pad(weights, [[0,0], [0,0], [0,0], [1,0]], constant_values=1.0)

        probs = tf.exp(logits) * weights
        probs_sum = tf.reduce_sum(probs, axis=-1, keepdims=True)
        probs = probs[:,:,:,1:]
        attn_feats = tf.matmul(probs, values) / probs_sum
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
                            scope=None):
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
            query_position_deltas = fully_connected(
                querys,
                num_head*size,
                dropout=dropout,
                is_training=is_training,
                scope="query_position_delta_projs")
            query_position_deltas = tf.stack(tf.split(query_position_deltas, num_head, axis=-1), axis=1)
           # query_position_deltas = tf.get_variable(
           #     'query_posit_deltas',
           #     shape=[num_head, size],
           #     dtype=tf.float32,
           #     initializer=tf.zeros_initializer(),
           #     trainable=(is_training != False))
           # query_position_deltas = tf.reshape(query_position_deltas, [1,num_head,1,size])
            query_position_translated_embeds = translate_position_embeds(
                query_position_embeds, query_position_deltas)

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
            logits += tf.matmul(query_position_translated_embeds, key_position_embeds, transpose_b=True)
        logits /= tf.sqrt(float(size))
        if masks != None:
            weights = tf.expand_dims(tf.cast(masks, tf.float32), axis=1)
        else:
            weights = tf.ones(tf.shape(logits))
        logits = logits * weights
        logits = tf.pad(logits, [[0,0], [0,0], [0,0], [1,0]])
        weights = tf.pad(weights, [[0,0], [0,0], [0,0], [1,0]], constant_values=1.0)

        probs = tf.nn.softmax(logits) * weights
        probs /= (tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-20)
        probs = probs[:,:,:,1:]
        attn_feats = tf.matmul(probs, values)
        if use_position:
            attn_posits = tf.matmul(probs, key_position_embeds)
            attn_posits = translate_position_embeds(attn_posits, query_position_embeds)
            attn_feats += fully_connected(
                attn_posits,
                size/num_head,
                is_training=is_training,
                scope="posit_projs")
        attn_feats = tf.concat(tf.unstack(attn_feats, axis=1), axis=-1)
        if single_query:
            attn_feats = tf.squeeze(attn_feats, 1)
    return attn_feats

def attention_write(querys,
                    values,
                    size=None,
                    num_head=1,
                    masks=None,
                    query_position_idxs=None,
                    key_position_idxs=None,
                    use_position=True,
                    use_write=True,
                    dropout=None,
                    is_training=True,
                    reuse=None,
                    scope=None):
    """ implements the attention mechanism

    querys: [batch_size x num_querys x dim]
    values: [batch_size x length x dim]
    masks: [batch_size x num_querys x length]
    """

    with tf.variable_scope(scope,
                           "attention",
                           [querys, values],
                           reuse=reuse) as sc:
        single_query = False
        if len(querys.get_shape()) == 2:
            single_query = True
            querys = tf.expand_dims(querys, 1)
            masks = tf.expand_dims(masks, 1) if masks != None else None
            if query_position_idxs != None:
                query_position_idxs = tf.expand_dims(query_position_idxs, 1)
        value_size = values.get_shape()[-1].value
        if size == None:
            size = value_size

        batch_size = tf.shape(querys)[0]
        query_len = tf.shape(querys)[1]
        key_len = tf.shape(values)[1]

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
            query_position_deltas = fully_connected(
                querys,
                num_head*size,
                dropout=dropout,
                is_training=is_training,
                scope="query_position_delta_projs")
            query_position_deltas = tf.stack(tf.split(query_position_deltas, num_head, axis=-1), axis=1)
           # query_position_deltas = tf.get_variable(
           #     'query_posit_deltas',
           #     shape=[num_head, size],
           #     dtype=tf.float32,
           #     initializer=tf.zeros_initializer(),
           #     trainable=(is_training != False))
           # query_position_deltas = tf.reshape(query_position_deltas, [1,num_head,1,size])
            query_position_translated_embeds = translate_position_embeds(
                query_position_embeds, query_position_deltas)

        key_content_embeds = fully_connected(
            values,
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

        value_projs = fully_connected(
            values,
            size,
            dropout=dropout,
            is_training=is_training,
            scope="value_projs")
        value_projs = tf.stack(tf.split(value_projs, num_head, axis=-1), axis=1)

        logits = tf.matmul(query_content_embeds, key_content_embeds, transpose_b=True)
        if use_position:
            logits += tf.matmul(query_position_translated_embeds, key_position_embeds, transpose_b=True)
        logits /= tf.sqrt(float(size))
        if masks != None:
            weights = tf.expand_dims(tf.cast(masks, tf.float32), axis=1)
        else:
            weights = tf.ones(tf.shape(logits))
        logits = logits * weights
        logits = tf.pad(logits, [[0,0], [0,0], [0,0], [1,0]])
        weights = tf.pad(weights, [[0,0], [0,0], [0,0], [1,0]], constant_values=1.0)

        probs = tf.nn.softmax(logits) * weights
        probs /= (tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-20)
        probs = probs[:,:,:,1:]
        attn_feats = tf.matmul(probs, value_projs)
        if use_position:
            attn_posits = tf.matmul(probs, key_position_embeds)
            attn_posits = translate_position_embeds(attn_posits, query_position_embeds)
            attn_feats += fully_connected(
                attn_posits,
                size/num_head,
                is_training=is_training,
                scope="posit_projs")
        attn_feats = tf.concat(tf.unstack(attn_feats, axis=1), axis=-1)
        if single_query:
            attn_feats = tf.squeeze(attn_feats, 1)

        if use_write:
            query_projs = fully_connected(
                querys,
                num_head*value_size,
                dropout=dropout,
                is_training=is_training,
                scope="query_write_projs")
            query_projs = tf.stack(tf.split(query_projs, num_head, axis=-1), axis=1)
            query_projs = tf.matmul(probs, query_projs, transpose_a=True)
            query_projs = tf.reduce_sum(query_projs, axis=1)
            values += query_projs
            values = tf.contrib.layers.layer_norm(values, begin_norm_axis=-1)

    return attn_feats, values

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

    position_idx = tf.cast(position_idx, tf.float32)
    size = tf.cast(size, tf.float32)
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

TransformerStruct = namedtuple('TransformerStruct', [
    'field_query_embeds',
    'field_key_embeds',
    'field_value_embeds',
    'posit_embeds',
    'token_embeds',
    'masks',
    'querys',
    'keys',
    'values',
    'encodes']
)

def transformer(tfstruct,
                num_layers,
                layer_size,
                extra_tfstruct=None,
                num_heads=8,
                attn_masks=None,
                dropout=None,
                is_training=True,
                reuse=None,
                scope=None):
    """Transformer encoder
    in the form of key-value
    args:
       tfstruct: {
           'field_query_embeds': tuple (batch_size x length x layer_size) * num_layers,
           'field_key_embeds': tuple (batch_size x length x layer_size) * num_layers,
           'field_value_embeds': tuple (batch_size x length x layer_size) * num_layers,
           'posit_embeds': batch_size x length x posit_size,
           'token_embeds': batch_size x length x embed_size,
           'masks': batch_size x length,
           'querys': tuple (batch_size x length x layer_size) * num_layers,
           'keys': tuple (batch_size x length x layer_size) * num_layers,
           'values': tuple (batch_size x length x layer_size) * num_layers,
           'encodes': batch_size x length x layer_size
       }
       num_layers: int
       layer_size: int
       extra_tfstruct: same structure, already encoded, or list/tuple
       num_heads: int
       attn_masks: batch_size x length x (length+extra_length)
    return:
       tfstruct: input tfstruct with missing slots filled
    """

    with tf.variable_scope(scope,
                           "Transformer",
                           flatten_structure([tfstruct, extra_tfstruct]),
                           reuse=reuse) as sc:

        batch_size = tf.shape(tfstruct.token_embeds)[0]
        length = tf.shape(tfstruct.token_embeds)[1]

        if extra_tfstruct is None:
            extra_length = 0
        elif isinstance(extra_tfstruct, TransformerStruct):
            extra_length = tf.shape(extra_tfstruct.masks)[1]
        else:
            extra_length = sum([tf.shape(e.masks)[1] for e in extra_tfstruct])

        token_encodes = fully_connected(
            tfstruct.token_embeds,
            layer_size,
            dropout=dropout,
            is_training=is_training,
            scope='enc_projs')

        if attn_masks == None:
            attn_masks = tf.ones([batch_size, length, length+extra_length], dtype=tf.bool)
        attn_masks = tf.logical_and(attn_masks,
            tf.logical_not(tf.pad(tf.eye(length, batch_shape=[batch_size], dtype=tf.bool), [[0,0],[0,0],[0,extra_length]])))

        query_list, key_list, value_list = [], [], []

        for i in range(num_layers):
            with tf.variable_scope("layer"+str(i)):
                encodes_normed = layer_norm(
                    token_encodes, begin_norm_axis=-1, is_training=is_training)
                querys = tf.concat([tfstruct.posit_embeds, encodes_normed], axis=-1)
                keys = querys
                values = encodes_normed
                querys = fully_connected(
                    querys,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="query_projs")
                querys += tfstruct.field_query_embeds[i]
                query_list.append(querys)
                keys = fully_connected(
                    keys,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="key_projs")
                keys += tfstruct.field_key_embeds[i]
                key_list.append(keys)
                values = GLU(
                    values,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="value_projs")
                values += tfstruct.field_value_embeds[i]
                value_list.append(values)
                if extra_tfstruct is None:
                    pass
                else:
                    def true_fn():
                        if isinstance(extra_tfstruct, TransformerStruct):
                            concat_keys = tf.concat(
                                [keys, extra_tfstruct.keys[i]],
                                axis=1)
                            concat_values = tf.concat(
                                [values, extra_tfstruct.values[i]],
                                axis=1)
                        else:
                            concat_keys = tf.concat(
                                [keys,] + [e.keys[i] for e in extra_tfstruct],
                                axis=1)
                            concat_values = tf.concat(
                                [values,] + [e.values[i] for e in extra_tfstruct],
                                axis=1)
                        return concat_keys, concat_values

                    keys, values = tf.cond(
                        tf.greater(extra_length, 0),
                        true_fn,
                        lambda: (keys, values))

                attn_feat = attention_simple(querys, keys, values,
                    num_heads=num_heads, masks=attn_masks, size=layer_size,
                    dropout=dropout, is_training=is_training)
                token_encodes += attn_feat
                encodes_normed = layer_norm(
                    token_encodes, begin_norm_axis=-1, is_training=is_training)
                token_encodes += MLP(
                    tf.concat([encodes_normed, attn_feat, tfstruct.token_embeds], axis=-1),
                    2,
                    layer_size,
                    layer_size,
                    activation_fn=tf.nn.relu,
                    dropout=dropout,
                    is_training=is_training)
        encodes_normed = layer_norm(
            token_encodes, begin_norm_axis=-1, is_training=is_training)
        tfstruct = TransformerStruct(
            field_query_embeds=tfstruct.field_query_embeds,
            field_key_embeds=tfstruct.field_key_embeds,
            field_value_embeds=tfstruct.field_value_embeds,
            posit_embeds=tfstruct.posit_embeds,
            token_embeds=tfstruct.token_embeds,
            masks=tfstruct.masks,
            querys=tuple(query_list),
            keys=tuple(key_list),
            values=tuple(value_list),
            encodes=encodes_normed,
        )
    return tfstruct

def get_tfstruct_si(tfstruct):
    """
    get shape invariants of tfstruct
    """
    tfstruct = TransformerStruct(
        field_query_embeds=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.field_query_embeds]),
        field_key_embeds=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.field_key_embeds]),
        field_value_embeds=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.field_value_embeds]),
        posit_embeds=tf.TensorShape(
            [tfstruct.posit_embeds.get_shape()[0],
             None,
             tfstruct.posit_embeds.get_shape()[2]]),
        token_embeds=tf.TensorShape(
            [tfstruct.token_embeds.get_shape()[0],
             None,
             tfstruct.token_embeds.get_shape()[2]]),
        masks=tf.TensorShape(
            [tfstruct.masks.get_shape()[0],
             None]),
        querys=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.querys]),
        keys=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.keys]),
        values=tuple(
            [tf.TensorShape([item.get_shape()[0], None, item.get_shape()[2]])
             for item in tfstruct.values]),
        encodes=tf.TensorShape(
            [tfstruct.encodes.get_shape()[0],
             None,
             tfstruct.encodes.get_shape()[2]]),
    )
    return tfstruct

def init_tfstruct(batch_size, embed_size, posit_size, layer_size, num_layers):
    """
    get a tfstruct with zero length
    """
    tfstruct = TransformerStruct(
        field_query_embeds=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        field_key_embeds=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        field_value_embeds=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        posit_embeds=tf.zeros([batch_size, 0, posit_size]),
        token_embeds=tf.zeros([batch_size, 0, embed_size]),
        masks=tf.zeros([batch_size, 0], dtype=tf.bool),
        querys=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        keys=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        values=(tf.zeros([batch_size, 0, layer_size]),)*num_layers,
        encodes=tf.zeros([batch_size, 0, layer_size]),
    )
    return tfstruct

def concat_tfstructs(tfstruct_list):
    """
    concat list of tfstructs: {
       'field_query_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'field_key_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'field_value_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'posit_embeds': batch_size x length x posit_size,
       'token_embeds': batch_size x length x embed_size,
       'masks': batch_size x length,
       'querys': tuple (batch_size x length x layer_size) * num_layers,
       'keys': tuple (batch_size x length x layer_size) * num_layers,
       'values': tuple (batch_size x length x layer_size) * num_layers,
       'encodes': batch_size x length x layer_size
    }
    return: concat tfstructs
    """
    tfstruct_list = list(filter(lambda i: not i is None, tfstruct_list))
    if len(tfstruct_list) == 0:
        return None
    elif len(tfstruct_list) == 1:
        return tfstruct_list[0]
    else:
        items = []
        for i in range(len(tfstruct_list[0])):
            feats = [tfstruct[i] for tfstruct in tfstruct_list]
            feats = list(filter(lambda f: not f is None, feats))
            if len(feats) == 0:
                items.append(None)
            elif len(feats) == 1:
                items.append(feats[0])
            else:
                if isinstance(feats[0], tf.Tensor):
                    items.append(tf.concat(feats, axis=1))
                elif isinstance(feats[0], tuple):
                    items.append(
                        tuple([tf.concat(list(fi), axis=1) for fi in zip(*feats)]))
                else:
                    items.append(none)
        tfstruct = TransformerStruct(*items)
        return tfstruct

def split_tfstructs(tfstruct, num_or_size_splits):
    """
    tfstruct: {
       'field_query_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'field_key_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'field_value_embeds': tuple (batch_size x length x layer_size) * num_layers,
       'posit_embeds': batch_size x length x posit_size,
       'token_embeds': batch_size x length x embed_size,
       'masks': batch_size x length,
       'querys': tuple (batch_size x length x layer_size) * num_layers,
       'keys': tuple (batch_size x length x layer_size) * num_layers,
       'values': tuple (batch_size x length x layer_size) * num_layers,
       'encodes': batch_size x length x layer_size
    }
    return: list of tfstructs
    """
    num_splits = len(num_or_size_splits) if isinstance(num_or_size_splits, list) else num_or_size_splits
    if tfstruct is None:
        return []
    elif num_splits == 1:
        return [tfstruct]
    else:
        tfstruct_list = []
        items_list = []
        for i in range(len(tfstruct)):
            if tfstruct[i] is None:
                items_list.append((None,)*num_splits)
            elif isinstance(tfstruct[i], tf.Tensor):
                items_list.append(
                    tuple(tf.split(tfstruct[i], num_or_size_splits, axis=1)))
            elif isinstance(tfstruct[i], tuple):
                items_list.append(
                    tuple(zip(*[tuple(tf.split(item, num_or_size_splits, axis=1))
                                for item in tfstruct[i]])))
            else:
                items_list.append((None,)*num_splits)
        items_list = list(zip(*items_list))
        for items in items_list:
            tfstruct_list.append(TransformerStruct(*items))
        return tfstruct_list

class CudnnLSTMCell(object):
    """Wrapper of tf.contrib.cudnn_rnn.CudnnLSTM"""

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

class AttentionCellWrapper(tf.contrib.rnn.RNNCell):
    """Wrapper for attention mechanism"""

    def __init__(self,
                 cell,
                 num_attention=2,
                 self_attention_idx=0,
                 use_copy=None,
                 use_coverage=None,
                 attention_fn=attention_simple,
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
                mask = tf.cast(state[2], tf.float32)
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
    """
    gather entries from state
    args:
        state: Tensor, TensorArray or possibly nested structure of them
        beam_parent: batch_beam_size
    """
    def gather_Tensor(state, beam_parent):
        state = tf.gather(state, beam_parent)
        return state

    def gather_TensorArray(state, beam_parent):
        size = state.size()
        last = state.read(size-1)
        new = tf.gather(last, beam_parent)
        state = state.write(size, new)
        return state

    if type(state) == tf.Tensor:
        state = tf.gather(state, beam_parent)
    elif type(state) == tf.TensorArray:
        size = state.size()
        last = state.read(size-1)
        new = tf.gather(last, beam_parent)
        state = state.write(size, new)
    else:
        l = [gather_state(s, beam_parent) for s in state]
        if type(state) == tuple:
            state = tuple(l)
        elif type(state) == list:
            state = l
        else:
            state = type(state)(*l)
    return state

def greedy_dec(length,
               initial_state,
               state_si_fn,
               cell,
               candidates_callback,
               start_embedding,
               start_id):
    """
    A greedy decoder.
    args:
        length: int
        initial_state:
        state_si: state shape invariants
        cell:
        candidates_callback:
            args:
                encodes: batch_size x output_dim
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: [batch_size x ]num_candidates
                logits: batch_size x num_candidates
        start_embedding: input_dim
        start_id: 0-dim or 1-dim tf.int32
    """

    flatten_state = flatten_structure(initial_state)
    for item in flatten_state:
        if isinstance(item, tf.Tensor):
            batch_size = tf.shape(item)[0]
            break
    inputs = tf.tile(tf.expand_dims(start_embedding, axis=0), [batch_size, 1])
    state = initial_state
    paths = tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1)
    if len(start_id.get_shape()) == 0:
        paths = tf.tile(paths, [batch_size, 1])
    else:
        paths = tf.tile(paths, [batch_size, 1, 1])
    scores = tf.zeros([batch_size])
    closed = tf.zeros([batch_size], dtype=tf.bool)

    def cond(inputs, state, paths, scores, closed):
        return tf.logical_not(tf.reduce_all(closed))

    def body(inputs, state, paths, scores, closed):
        """
        args:
            inputs: batch_size x input_dim
            state:
            paths: batch_size x current_length [x word_len]
            scores: batch_size
            closed: batch_size
        """
        not_closed = tf.logical_not(closed)
        cur_len = tf.shape(paths)[1]
        outputs, state = cell(inputs, state)
        candidate_embeds, candidate_ids, candidate_masks, logits = candidates_callback(outputs)
        log_probs = logits + tf.log(tf.cast(candidate_masks, tf.float32))
        # select best 1
        indices = tf.argmax(log_probs, 1, output_type=tf.int32)
        indices *= tf.cast(not_closed, tf.int32)
        batch_indices = tf.stack([tf.range(tf.shape(indices)[0], dtype=tf.int32), indices], axis=1)
        closing = tf.logical_and(tf.equal(indices, 0), not_closed)
        # update scores
        new_score = tf.gather_nd(log_probs, batch_indices)
        scores += new_score * tf.cast(not_closed, tf.float32)
        scores *= tf.pow(tf.cast(cur_len, tf.float32), -tf.cast(closing, tf.float32))
        # update paths and inputs
        if len(candidate_embeds.get_shape()) == 2:
            new_ids = tf.gather(candidate_ids, indices)
            inputs = tf.gather(candidate_embeds, indices)
        elif len(candidate_embeds.get_shape()) == 3:
            new_ids = tf.gather_nd(candidate_ids, batch_indices)
            inputs = tf.gather_nd(candidate_embeds, batch_indices)
        if len(start_id.get_shape()) == 0:
            new_ids *= tf.cast(not_closed, tf.int32)
        else:
            new_ids *= tf.expand_dims(tf.cast(not_closed, tf.int32), axis=1)
        new_ids = tf.expand_dims(new_ids, axis=1)
        paths = tf.concat([paths, new_ids], axis=1)
        # update closed
        closed = tf.logical_or(closed, closing)
        return inputs, state, paths, scores, closed

    # shape_invariants
    inputs_si = inputs.get_shape()
    state_si = state_si_fn(state)
    scores_si = scores.get_shape()
    closed_si = closed.get_shape()
    if len(start_id.get_shape()) == 0:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None])
    else:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None, paths.get_shape()[2]])

    inputs, state, paths, scores, closed = tf.while_loop(
        cond, body, (inputs, state, paths, scores, closed),
        shape_invariants=(inputs_si, state_si, paths_si, scores_si, closed_si),
        back_prop=False,
        parallel_iterations=64,
        maximum_iterations=length)

    final_seqs = tf.expand_dims(paths, axis=1)
    final_scores = tf.expand_dims(scores, axis=1)

    return final_seqs, final_scores

def stochastic_dec(length,
                   initial_state,
                   state_si_fn,
                   cell,
                   candidates_callback,
                   start_embedding,
                   start_id,
                   num_candidates=1):
    """
    A stochastic decoder.
    args:
        length: int
        initial_state:
        state_si: state shape invariants
        cell:
        candidates_callback:
            args:
                encodes: batch_size x output_dim
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: [batch_size x ]num_candidates
                logits: batch_size x num_candidates
        start_embedding: input_dim
        start_id: 0-dim or 1-dim tf.int32
    """

    flatten_state = flatten_structure(initial_state)
    for item in flatten_state:
        if isinstance(item, tf.Tensor):
            batch_size = tf.shape(item)[0]
            break
    inputs = tf.tile(tf.expand_dims(start_embedding, axis=0), [batch_size*num_candidates, 1])
    beam_parent = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), [1, num_candidates]),
        [batch_size*num_candidates])
    state = gather_state(initial_state, beam_parent)
    paths = tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1)
    if len(start_id.get_shape()) == 0:
        paths = tf.tile(paths, [batch_size*num_candidates, 1])
    else:
        paths = tf.tile(paths, [batch_size*num_candidates, 1, 1])
    scores = tf.zeros([batch_size*num_candidates])
    closed = tf.zeros([batch_size*num_candidates], dtype=tf.bool)

    def cond(inputs, state, paths, scores, closed):
        return tf.logical_not(tf.reduce_all(closed))

    def body(inputs, state, paths, scores, closed):
        """
        args:
            inputs: batch_size x input_dim
            state:
            paths: batch_size x current_length [x word_len]
            scores: batch_size
            closed_paths: batch_size x current_length x length [x word_len]
            closed_scores: batch_size x current_length
        """
        not_closed = tf.logical_not(closed)
        cur_len = tf.shape(paths)[1]
        outputs, state = cell(inputs, state)
        candidate_embeds, candidate_ids, candidate_masks, logits = candidates_callback(outputs)
        log_probs = logits + tf.log(tf.cast(candidate_masks, tf.float32))
        # random sample
        indices = tf.squeeze(tf.random.categorical(log_probs, 1, dtype=tf.int32), [1])
        indices *= tf.cast(not_closed, tf.int32)
        batch_indices = tf.stack([tf.range(tf.shape(indices)[0], dtype=tf.int32), indices], axis=1)
        closing = tf.logical_and(tf.equal(indices, 0), not_closed)
        # update scores
        new_score = tf.gather_nd(log_probs, batch_indices)
        scores += new_score * tf.cast(not_closed, tf.float32)
        scores *= tf.pow(tf.cast(cur_len, tf.float32), -tf.cast(closing, tf.float32))
        # update paths and inputs
        if len(candidate_embeds.get_shape()) == 2:
            new_ids = tf.gather(candidate_ids, indices)
            inputs = tf.gather(candidate_embeds, indices)
        elif len(candidate_embeds.get_shape()) == 3:
            new_ids = tf.gather_nd(candidate_ids, batch_indices)
            inputs = tf.gather_nd(candidate_embeds, batch_indices)
        if len(start_id.get_shape()) == 0:
            new_ids *= tf.cast(not_closed, tf.int32)
        else:
            new_ids *= tf.expand_dims(tf.cast(not_closed, tf.int32), axis=1)
        new_ids = tf.expand_dims(new_ids, axis=1)
        paths = tf.concat([paths, new_ids], axis=1)
        # update closed
        closed = tf.logical_or(closed, closing)
        return inputs, state, paths, scores, closed

    # shape_invariants
    inputs_si = inputs.get_shape()
    state_si = state_si_fn(state)
    scores_si = scores.get_shape()
    closed_si = closed.get_shape()
    if len(start_id.get_shape()) == 0:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None])
    else:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None, paths.get_shape()[2]])

    inputs, state, paths, scores, closed = tf.while_loop(
        cond, body, (inputs, state, paths, scores, closed),
        shape_invariants=(inputs_si, state_si, paths_si, scores_si, closed_si),
        back_prop=False,
        parallel_iterations=64,
        maximum_iterations=length)

    if len(start_id.get_shape()) == 0:
        final_seqs = tf.reshape(paths, [batch_size, num_candidates, tf.shape(paths)[1]])
    else:
        final_seqs = tf.reshape(paths, [batch_size, num_candidates, tf.shape(paths)[1], tf.shape(paths)[2]])
    final_scores = tf.reshape(scores, [batch_size, num_candidates])

    return final_seqs, final_scores

def beam_dec(length,
             initial_state,
             state_si_fn,
             cell,
             candidates_callback,
             start_embedding,
             start_id,
             beam_size=16,
             num_candidates=1):
    """
    A beam decoder.
    args:
        length: int
        initial_state:
        state_si: state shape invariants
        cell:
        candidates_callback:
            args:
                encodes: batch_size x output_dim
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: [batch_size x ]num_candidates
                logits: batch_size x num_candidates
        start_embedding: input_dim
        start_id: 0-dim or 1-dim tf.int32
    """

    flatten_state = flatten_structure(initial_state)
    for item in flatten_state:
        if isinstance(item, tf.Tensor):
            batch_size = tf.shape(item)[0]
            break
    inputs = tf.tile(tf.expand_dims(start_embedding, axis=0), [batch_size*beam_size, 1])
    beam_parent = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), [1, beam_size]),
        [batch_size*beam_size])
    state = gather_state(initial_state, beam_parent)
    paths = tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1)
    if len(start_id.get_shape()) == 0:
        paths = tf.tile(paths, [batch_size*beam_size, 1])
        closed_paths = tf.zeros([batch_size*beam_size, 0, length+1], dtype=tf.int32)
        end_ids = tf.tile(
            tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1), [batch_size*beam_size,1])
    else:
        paths = tf.tile(paths, [batch_size*beam_size, 1, 1])
        closed_paths = tf.zeros([batch_size*beam_size, 0, length+1, tf.shape(start_id)[0]], dtype=tf.int32)
        end_ids = tf.tile(
            tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1), [batch_size*beam_size,1,1])
    scores = tf.concat(
        [tf.ones([batch_size, 1]), tf.zeros([batch_size, beam_size-1])], axis=1)
    scores = tf.reshape(tf.log(scores), [batch_size*beam_size])
    closed_scores = tf.zeros([batch_size*beam_size, 0])

    def cond(inputs, state, paths, scores, closed_paths, closed_scores):
        return tf.constant(True, dtype=tf.bool)

    def body(inputs, state, paths, scores, closed_paths, closed_scores):
        """
        args:
            inputs: batch_size x input_dim
            state:
            paths: batch_size x current_length [x word_len]
            scores: batch_size
            closed_paths: batch_size x current_length x (length+1) [x word_len]
            closed_scores: batch_size x current_length
        """
        cur_len = tf.shape(paths)[1]

        # iter
        outputs, state = cell(inputs, state)
        candidate_embeds, candidate_ids, candidate_masks, logits = candidates_callback(outputs)
        vocab_size = tf.shape(logits)[1]
        log_probs = logits + tf.log(tf.cast(candidate_masks, tf.float32))

        # closing mask
        closing_masks = tf.math.in_top_k(
            log_probs,
            tf.zeros([batch_size*beam_size], dtype=tf.int32),
            beam_size)

        # closed scores
        closing_scores = (log_probs[:,0] + scores) / tf.cast(cur_len, tf.float32)
        closing_scores += tf.log(tf.cast(closing_masks, tf.float32))
        closed_scores = tf.concat([closed_scores, tf.expand_dims(closing_scores, axis=1)], axis=1)

        # closed paths
        closing_paths = tf.concat([paths, end_ids], axis=1)
        if len(paths.get_shape()) == 2:
            closing_paths = tf.pad(closing_paths, [[0,0],[0,length-cur_len]])
        elif len(paths.get_shape()) == 3:
            closing_paths = tf.pad(closing_paths, [[0,0],[0,length-cur_len],[0,0]])
        closed_paths = tf.concat([closed_paths, tf.expand_dims(closing_paths, axis=1)], axis=1)

        # open
        open_scores = log_probs[:, 1:] + tf.expand_dims(scores, axis=1)
        open_scores = tf.reshape(open_scores, [batch_size, -1])
        top_scores, top_indices = tf.math.top_k(open_scores, beam_size)
        scores = tf.reshape(top_scores, [-1])

        # gather beam parent
        beam_parent = tf.floor_div(top_indices, (vocab_size-1))
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size)*beam_size, 1)+beam_parent, [-1])
        state = gather_state(state, beam_parent)
        paths = tf.gather(paths, beam_parent)

        # next
        indices = tf.floormod(top_indices, (vocab_size-1))
        indices = tf.reshape(indices, [-1])
        batch_indices = tf.stack([tf.range(tf.shape(indices)[0], dtype=tf.int32), indices], axis=1)
        if len(candidate_embeds.get_shape()) == 2:
            open_candidate_ids = candidate_ids[1:]
            open_candidate_embeds = candidate_embeds[1:]
            new_ids = tf.expand_dims(tf.gather(open_candidate_ids, indices), axis=1)
            inputs = tf.gather(open_candidate_embeds, indices)
        elif len(candidate_embeds.get_shape()) == 3:
            open_candidate_ids = tf.gather(candidate_ids[:,1:], beam_parent)
            open_candidate_embeds = tf.gather(candidate_embeds[:,1:], beam_parent)
            new_ids = tf.expand_dims(tf.gather_nd(open_candidate_ids, batch_indices), axis=1)
            inputs = tf.gather_nd(open_candidate_embeds, batch_indices)
        paths = tf.concat([paths, new_ids], axis=1)

        return inputs, state, paths, scores, closed_paths, closed_scores

    # shape_invariants
    inputs_si = inputs.get_shape()
    state_si = state_si_fn(state)
    scores_si = scores.get_shape()
    closed_scores_si = tf.TensorShape(
        [closed_scores.get_shape()[0], None])
    if len(start_id.get_shape()) == 0:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None])
        closed_paths_si = tf.TensorShape(
            [closed_paths.get_shape()[0], None, closed_paths.get_shape()[2]])
    else:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None, paths.get_shape()[2]])
        closed_paths_si = tf.TensorShape(
            [closed_paths.get_shape()[0], None, closed_paths.get_shape()[2],
             closed_paths.get_shape()[3]])

    inputs, state, paths, scores, closed_paths, closed_scores = tf.while_loop(
        cond, body, (inputs, state, paths, scores, closed_paths, closed_scores),
        shape_invariants=(inputs_si, state_si, paths_si, scores_si, closed_paths_si, closed_scores_si),
        back_prop=False,
        parallel_iterations=4,
        maximum_iterations=length)

    closed_scores = tf.reshape(closed_scores, [batch_size, beam_size*length])
    if len(start_id.get_shape()) == 0:
        closed_paths = tf.reshape(
            closed_paths, [batch_size, beam_size*length, length+1])
    else:
        closed_paths = tf.reshape(
            closed_paths, [batch_size, beam_size*length, length+1, tf.shape(closed_paths)[3]])
    best_scores, best_indices = tf.nn.top_k(closed_scores, num_candidates)
    batch_best_indices = tf.stack(
        [tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), [1, num_candidates]),
         best_indices],
        axis=2)
    best_paths = tf.gather_nd(closed_paths, batch_best_indices)

    return best_paths, best_scores

def stochastic_beam_dec(length,
                        initial_state,
                        state_si_fn,
                        cell,
                        candidates_callback,
                        start_embedding,
                        start_id,
                        beam_size=16,
                        num_candidates=1,
                        cutoff_size=128,
                        gamma=4):
    """
    A stochastic beam decoder.
    args:
        length: int
        initial_state:
        state_si: state shape invariants
        cell:
        candidates_callback:
            args:
                encodes: batch_size x output_dim
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: [batch_size x ]num_candidates
                logits: batch_size x num_candidates
        start_embedding: input_dim
        start_id: 0-dim or 1-dim tf.int32
    """

    flatten_state = flatten_structure(initial_state)
    for item in flatten_state:
        if isinstance(item, tf.Tensor):
            batch_size = tf.shape(item)[0]
            break
    inputs = tf.tile(tf.expand_dims(start_embedding, axis=0), [batch_size*beam_size, 1])
    beam_parent = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), [1, beam_size]),
        [batch_size*beam_size])
    state = gather_state(initial_state, beam_parent)
    paths = tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1)
    if len(start_id.get_shape()) == 0:
        paths = tf.tile(paths, [batch_size*beam_size, 1])
        closed_paths = tf.zeros([batch_size*beam_size, 0, length+1], dtype=tf.int32)
        end_ids = tf.tile(
            tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1), [batch_size*beam_size,1])
    else:
        paths = tf.tile(paths, [batch_size*beam_size, 1, 1])
        closed_paths = tf.zeros([batch_size*beam_size, 0, length+1, tf.shape(start_id)[0]], dtype=tf.int32)
        end_ids = tf.tile(
            tf.expand_dims(tf.expand_dims(start_id, axis=0), axis=1), [batch_size*beam_size,1,1])
    scores = tf.concat(
        [tf.ones([batch_size, 1]), tf.zeros([batch_size, beam_size-1])], axis=1)
    scores = tf.reshape(tf.log(scores), [batch_size*beam_size])
    closed_scores = tf.zeros([batch_size*beam_size, 0])

    def cond(inputs, state, paths, scores, closed_paths, closed_scores):
        return tf.constant(True, dtype=tf.bool)

    def body(inputs, state, paths, scores, closed_paths, closed_scores):
        """
        args:
            inputs: batch_size x input_dim
            state:
            paths: batch_size x current_length [x word_len]
            scores: batch_size
            closed_paths: batch_size x current_length x (length+1) [x word_len]
            closed_scores: batch_size x current_length
        """
        cur_len = tf.shape(paths)[1]
        cur_len_fp32 = tf.cast(cur_len, tf.float32)
        beta = 1.0 / (1.0 - math.exp(-1.0/gamma))
        alpha = beta * (1.0 - tf.exp((-cur_len_fp32) / gamma)) / cur_len_fp32

        # iter
        outputs, state = cell(inputs, state)
        candidate_embeds, candidate_ids, candidate_masks, logits = candidates_callback(outputs)
        vocab_size = tf.shape(logits)[1]
        log_probs = logits + tf.log(tf.cast(candidate_masks, tf.float32))

        # closing mask
        closing_masks = tf.math.in_top_k(
            log_probs,
            tf.zeros([batch_size*beam_size], dtype=tf.int32),
            cutoff_size)

        # closed scores
        closing_scores = (log_probs[:,0] + scores) * alpha
        closing_scores += tf.log(tf.cast(closing_masks, tf.float32))
        closed_scores = tf.concat([closed_scores, tf.expand_dims(closing_scores, axis=1)], axis=1)

        # closed paths
        closing_paths = tf.concat([paths, end_ids], axis=1)
        if len(paths.get_shape()) == 2:
            closing_paths = tf.pad(closing_paths, [[0,0],[0,length-cur_len]])
        elif len(paths.get_shape()) == 3:
            closing_paths = tf.pad(closing_paths, [[0,0],[0,length-cur_len],[0,0]])
        closed_paths = tf.concat([closed_paths, tf.expand_dims(closing_paths, axis=1)], axis=1)

        # open
        open_scores = log_probs[:, 1:] + tf.expand_dims(scores, axis=1)
        open_scores = tf.reshape(open_scores, [batch_size, -1])
        sample_indices = tf.random.categorical(
            open_scores * alpha,
            beam_size, dtype=tf.int32)
        batch_indices = tf.stack(
            [tf.tile(
                tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                [1, beam_size]), sample_indices],
            axis=2)
        sample_scores = tf.gather_nd(open_scores, batch_indices)
        scores = tf.reshape(sample_scores, [-1])

        # gather beam parent
        beam_parent = tf.floor_div(sample_indices, (vocab_size-1))
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size)*beam_size, 1)+beam_parent, [-1])
        state = gather_state(state, beam_parent)
        paths = tf.gather(paths, beam_parent)

        # next
        indices = tf.floormod(sample_indices, (vocab_size-1))
        indices = tf.reshape(indices, [-1])
        batch_indices = tf.stack([tf.range(tf.shape(indices)[0], dtype=tf.int32), indices], axis=1)
        if len(candidate_embeds.get_shape()) == 2:
            open_candidate_ids = candidate_ids[1:]
            open_candidate_embeds = candidate_embeds[1:]
            new_ids = tf.expand_dims(tf.gather(open_candidate_ids, indices), axis=1)
            inputs = tf.gather(open_candidate_embeds, indices)
        elif len(candidate_embeds.get_shape()) == 3:
            open_candidate_ids = tf.gather(candidate_ids[:,1:], beam_parent)
            open_candidate_embeds = tf.gather(candidate_embeds[:,1:], beam_parent)
            new_ids = tf.expand_dims(tf.gather_nd(open_candidate_ids, batch_indices), axis=1)
            inputs = tf.gather_nd(open_candidate_embeds, batch_indices)
        paths = tf.concat([paths, new_ids], axis=1)

        return inputs, state, paths, scores, closed_paths, closed_scores

    # shape_invariants
    inputs_si = inputs.get_shape()
    state_si = state_si_fn(state)
    scores_si = scores.get_shape()
    closed_scores_si = tf.TensorShape(
        [closed_scores.get_shape()[0], None])
    if len(start_id.get_shape()) == 0:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None])
        closed_paths_si = tf.TensorShape(
            [closed_paths.get_shape()[0], None, closed_paths.get_shape()[2]])
    else:
        paths_si = tf.TensorShape(
            [paths.get_shape()[0], None, paths.get_shape()[2]])
        closed_paths_si = tf.TensorShape(
            [closed_paths.get_shape()[0], None, closed_paths.get_shape()[2],
             closed_paths.get_shape()[3]])

    inputs, state, paths, scores, closed_paths, closed_scores = tf.while_loop(
        cond, body, (inputs, state, paths, scores, closed_paths, closed_scores),
        shape_invariants=(inputs_si, state_si, paths_si, scores_si, closed_paths_si, closed_scores_si),
        back_prop=False,
        parallel_iterations=4,
        maximum_iterations=length)

    closed_scores = tf.reshape(closed_scores, [batch_size, beam_size*length])
    if len(start_id.get_shape()) == 0:
        closed_paths = tf.reshape(
            closed_paths, [batch_size, beam_size*length, length+1])
    else:
        closed_paths = tf.reshape(
            closed_paths, [batch_size, beam_size*length, length+1, tf.shape(closed_paths)[3]])
    chosen_indices = tf.random.categorical(closed_scores, num_candidates, dtype=tf.int32)
    batch_chosen_indices = tf.stack(
        [tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1), [1, num_candidates]),
         chosen_indices],
        axis=2)
    chosen_paths = tf.gather_nd(closed_paths, batch_chosen_indices)
    chosen_scores = tf.gather_nd(closed_scores, batch_chosen_indices)

    return chosen_paths, chosen_scores


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
    padded_seq_masks = tf.not_equal(padded_seqs, 0)
    padded_seg_masks1 = tf.logical_or(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_and(padded_segs, padded_seg_masks1)
    padded_seg_masks2 = tf.logical_xor(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_or(padded_segs, padded_seg_masks2)

    num_words = tf.maximum(
        tf.reduce_sum(tf.cast(padded_segs, tf.int32), axis=1)-1, 0)
    max_num_word = tf.maximum(tf.reduce_max(num_words), 1)
    idx = tf.range(max_length+1, dtype=tf.int32)

    def get_idx(args):
        padded_seg, num_word = args
        valid_idx = tf.boolean_mask(idx, padded_seg)
        pad_num = max_num_word-num_word
        start = tf.pad(valid_idx[:-1], [[0, pad_num]])
        length = tf.pad(valid_idx[1:] - valid_idx[:-1], [[0, pad_num]])
        return start, length

    starts, lengths = tf.map_fn(
        get_idx,
        (padded_segs, num_words),
        (tf.int32, tf.int32),
        parallel_iterations=128,
        back_prop=False,
        swap_memory=True)

    results = []

    segmented_seqs = slice_fragments(
        tf.cast(tf.expand_dims(seqs, axis=-1), tf.float32),
        starts,
        lengths)
    segmented_seqs = tf.cast(tf.squeeze(segmented_seqs, axis=-1), tf.int32)
    results.append(segmented_seqs)

    if get_idxs:
        idx_starts = tf.cast(tf.logical_not(tf.sequence_mask(starts, max_length)), tf.int32)
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
    num_chars = tf.reduce_sum(tf.cast(masks, tf.int32), axis=[1,2])
    max_num_char = tf.reduce_max(num_chars)

    def stitch(segmented_seq):
        mask = tf.greater(segmented_seq, 0)
        seq = tf.boolean_mask(
            segmented_seq,
            mask)
        num_char = tf.shape(seq)[0]
        seq = tf.pad(seq, [[0,max_num_char-num_char]])

        word_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        seg_pos, _ = tf.unique(tf.math.cumsum(word_len) - 1)
        seg = tf.reduce_sum(tf.one_hot(seg_pos, max_num_char), axis=0)
        return seq, seg

    seqs, segs = tf.map_fn(
        stitch,
        segmented_seqs,
        (tf.int32, tf.float32),
        parallel_iterations=128,
        back_prop=False,
        swap_memory=True)
    segs = segs[:,:-1]
    return seqs, segs

def match_vector(x, y):
    """
    args:
        x: batch_size x length1 x dim or length1 x dim
        y: batch_size x length2 x dim or length2 x dim
    return:
        match_matrix: batch_size x length1 x length2 or lengh1 x length2, bool
    """
    match_matrix = tf.equal(
        tf.expand_dims(x, axis=-2),
        tf.expand_dims(y, axis=-3))
    match_matrix = tf.reduce_all(match_matrix, axis=-1)
    return match_matrix

def unique_vector(x):
    """
    args:
        x: num_vectors x vector_dim
    returns:
        unique elems of x
        indices in x of the unique elems
    """
    x_shape=tf.shape(x) #(3,2)
    x1=tf.tile(x,[1,x_shape[0]]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2=tf.tile(x,[x_shape[0],1]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2=tf.reshape(x1,[x_shape[0]*x_shape[0],x_shape[1]])
    x2_2=tf.reshape(x2,[x_shape[0]*x_shape[0],x_shape[1]])
    cond=tf.reduce_all(tf.equal(x1_2,x2_2),axis=1)
    cond=tf.reshape(cond,[x_shape[0],x_shape[0]]) #reshaping cond to match x1_2 & x2_2
    cond_shape=tf.shape(cond)
    cond_cast=tf.cast(cond,tf.int32) #convertin condition boolean to int
    cond_zeros=tf.zeros(cond_shape,tf.int32) #replicating condition tensor into all 0's

    #CREATING RANGE TENSOR
    r=tf.range(x_shape[0])
    r=tf.add(tf.tile(r,[x_shape[0]]),1)
    r=tf.reshape(r,[x_shape[0],x_shape[0]])

    #converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape,tf.int32),x_shape[0]+1)
    f2 =tf.ones(cond_shape,tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast,cond_zeros),f1,f2) #if false make it max_index+1 else keep it 1

    #multiply range with new int boolean mask
    r_cond_mul=tf.multiply(r,cond_cast2)
    r_cond_mul2=tf.reduce_min(r_cond_mul,axis=1)
    r_cond_mul3,unique_idx=tf.unique(r_cond_mul2)
    r_cond_mul4=tf.subtract(r_cond_mul3,1)

    #get actual values from unique indexes
    unique_x=tf.gather(x,r_cond_mul4)

    return unique_x, r_cond_mul4

def pad_vectors(vector_list):
    """
    pad a list of vectors to same size in last dim
    """
    if len(vector_list) <= 1:
        return vector_list
    else:
        max_size = tf.reduce_max(tf.stack([tf.shape(v)[-1] for v in vector_list], axis=0))
        for i, v in enumerate(vector_list):
            num_dim = len(v.get_shape())
            vector_list[i] = tf.cond(
                tf.less(tf.shape(v)[-1], max_size),
                lambda: tf.pad(v, [[0,0],]*(num_dim-1)+[[0,max_size-tf.shape(v)[-1]]]),
                lambda: v)
        return vector_list

def flatten_structure(struct):
    """flatten a possibly nested struct into a list"""
    flat_list = []
    if struct is None:
        pass
    elif isinstance(struct, str):
        flat_list.append(struct)
    elif isinstance(struct, tf.Tensor):
        flat_list.append(struct)
    elif isinstance(struct, tf.TensorArray):
        flat_list.append(struct)
    elif isinstance(struct, Iterable):
        for item in struct:
            flat_list.extend(flatten_structure(item))
    else:
        flat_list.append(struct)
    return flat_list