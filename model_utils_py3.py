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
        if dropout == None:
            outputs = inputs
        else:
            outputs = tf.cond(
                tf.cast(is_training, tf.bool),
                lambda: tf.nn.dropout(inputs, rate=dropout),
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

        querys = tf.stack(tf.split(querys, num_head, axis=-1), axis=1)

        keys = tf.stack(tf.split(keys, num_head, axis=-1), axis=1)

        values = tf.stack(tf.split(values, num_head, axis=-1), axis=1)
        
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

def transformer(field_embeds,
                posit_embeds,
                token_embeds,
                num_layers,
                layer_size,
                extra_field_embeds=None,
                extra_posit_embeds=None,
                extra_encodes=None,
                num_head=8,
                masks=None,
                dropout=None,
                is_training=True,
                reuse=None,
                scope=None):
    """Transformer encoder
       in the form of key-value
       args:
           field_embeds: batch_size x length x num_layers x 2*layer_size
           posit_embeds: batch_size x length x layer_size
           token_embeds: batch_size x length x layer_size
       return:
           encodes: batch_size x length x (num_layers+1) x layer_size
    """

    with tf.variable_scope(scope,
                           "Transformer",
                           [field_embeds, posit_embeds, token_embeds],
                           reuse=reuse) as sc:

        trainable = (is_training != False)
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if trainable:
            collections.append(tf.GraphKeys.WEIGHTS)

        batch_size = tf.shape(token_embeds)[0]
        length = tf.shape(token_embeds)[1]
        token_encodes = fully_connected(
            token_embeds,
            layer_size,
            dropout=dropout,
            is_training=is_training,
            scope='enc_projs')
        if masks != None:
            if len(masks.get_shape()) != 3:
                masks = tf.tile(tf.expand_dims(masks, 1), [1,length,1])
        else:
            masks = tf.ones([batch_size, length, length], dtype=tf.bool)
        masks = tf.logical_and(masks,
            tf.logical_not(tf.eye(length, batch_shape=[batch_size], dtype=tf.bool)))
        field_query_embeds, field_key_embeds, field_value_embeds = tf.split(field_embeds, 3, axis=-1)
        field_query_embeds = tf.unstack(field_query_embeds, axis=2)
        field_key_embeds = tf.unstack(field_key_embeds, axis=2)
        field_value_embeds = tf.unstack(field_value_embeds, axis=2)
        if not extra_encodes is None:
            extra_encodes_list = tf.unstack(extra_encodes, axis=2)
            extra_field_query_embeds, extra_field_key_embeds, extra_field_value_embeds = tf.split(extra_field_embeds, 3, axis=-1)
            extra_field_key_embeds = tf.unstack(extra_field_key_embeds, axis=2)
            extra_field_value_embeds = tf.unstack(extra_field_value_embeds, axis=2)
        encodes_list = []
        for i in range(num_layers):
            with tf.variable_scope("layer"+str(i)):
                encodes_normed = layer_norm(
                    token_encodes, begin_norm_axis=-1, is_training=is_training)
                encodes_list.append(encodes_normed)
                querys = tf.concat([posit_embeds, encodes_normed], axis=-1)
                keys = querys
                values = encodes_normed
                querys = fully_connected(
                    querys,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="query_projs")
                querys += field_query_embeds[i]
                if not extra_encodes is None:
                    field_key_embeds[i] = tf.concat([field_key_embeds[i], extra_field_key_embeds[i]], axis=1)
                    field_value_embeds[i] = tf.concat([field_value_embeds[i], extra_field_value_embeds[i]], axis=1)
                    keys = tf.concat([keys, tf.concat([extra_posit_embeds, extra_encodes_list[i]], axis=-1)], axis=1)
                    values = tf.concat([values, extra_encodes_list[i]], axis=1)
                keys = fully_connected(
                    keys,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="key_projs")
                keys += field_key_embeds[i]
                values = GLU(
                    values,
                    layer_size,
                    dropout=dropout,
                    is_training=is_training,
                    scope="value_projs")
                values += field_value_embeds[i]
                attn_feat = attention_simple(querys, keys, values,
                    num_head=num_head, masks=masks, size=layer_size,
                    dropout=dropout, is_training=is_training)
                token_encodes += attn_feat
                encodes_normed = layer_norm(
                    token_encodes, begin_norm_axis=-1, is_training=is_training)
                token_encodes += MLP(
                    tf.concat([encodes_normed, attn_feat, token_embeds], axis=-1),
                    2,
                    layer_size,
                    layer_size,
                    activation_fn=tf.nn.relu,
                    dropout=dropout,
                    is_training=is_training)
        encodes_normed = layer_norm(
            token_encodes, begin_norm_axis=-1, is_training=is_training)
        encodes_list.append(encodes_normed)
        encodes = tf.stack(encodes_list, axis=2)
    return encodes


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

class AttentionCell(object):
    """
    Attention Cell for sequence generation

    inputs: batch_size x [input_length] x input_dim
    state: tuple(decoder_inputs, encodes, masks)
        decoder_inputs: TensorArray with last item be the history inputs with batch_size x length x input_dim
        field_embeds: batch_size x 2 x 3*dim
    """
    

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
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]
            trainable=(self.is_training != False)
            if trainable:
                collections.append(tf.GraphKeys.WEIGHTS)

            batch_size = tf.shape(inputs)[0]
            input_dim = inputs.get_shape()[-1].value
            decoder_inputs = state[0]
            field_embeds = state[1]
            to_squeeze = False
            if inputs.shape.ndims == 2:
                to_squeeze = True
                idx = decoder_inputs.size()
                decoder_inputs_tensor = tf.cond(
                    tf.greater(idx, 0),
                    lambda: tf.concat([tf.reshape(decoder_inputs.read(idx-1), [batch_size, -1, input_dim]),
                        tf.expand_dims(inputs, 1)], axis=1),
                    lambda: tf.expand_dims(inputs, 1))
                state = tuple([decoder_inputs.write(idx, decoder_inputs_tensor)] + list(state[1:]))
                length = 1
            else:
                idx = decoder_inputs.size()
                decoder_inputs_tensor = tf.cond(
                    tf.greater(idx, 0),
                    lambda: tf.concat([tf.reshape(decoder_inputs.read(idx-1), [batch_size, -1, input_dim]),
                        inputs], axis=1),
                    lambda: inputs)
                decoder_inputs_tensor = tf.reshape(decoder_inputs_tensor, [batch_size, -1, input_dim])
                state = tuple([decoder_inputs.write(idx, decoder_inputs_tensor)] + list(state[1:]))
                length = tf.shape(inputs)[1]
            start_idx = tf.shape(decoder_inputs_tensor)[1] - length
            end_idx = tf.shape(decoder_inputs_tensor)[1]

            field_embeds_list = []
            posit_embeds_list = []
            token_embeds_list = []

            # decoder inputs part
            posit_embeds = embed_position(
                tf.tile(tf.expand_dims(tf.range(end_idx), 0), [batch_size, 1]),
                int(self.size/4))
            posit_embeds_list.append(posit_embeds)
            field_embeds_list.append(tf.tile(tf.expand_dims(field_embeds, axis=1), [1, end_idx, 1, 1]))
            token_embeds_list.append(inputs)

            # decoder output part
            posit_embeds = embed_position(
                tf.tile(tf.expand_dims(tf.range(start_idx+1, end_idx+1), 0), [batch_size, 1]),
                int(self.size/4))
            posit_embeds_list.append(posit_embeds)
            field_embeds_list.append(tf.tile(tf.expand_dims(field_embeds, axis=1), [1, length, 1, 1]))
            token_embeds_list.append(tf.zeros([batch_size, length, input_dim]))

            # prepare masks
            attn_masks = tf.concat(
                [tf.sequence_mask(
                     tf.tile(tf.expand_dims(tf.range(0, end_idx),0), [batch_size,1]),
                     maxlen=end_idx+length),
                 tf.sequence_mask(
                     tf.tile(tf.expand_dims(tf.range(start_idx, end_idx)+1,0), [batch_size,1]),
                     maxlen=end_idx+length)],
                axis=1)

            outputs = transformer(
                tf.concat(field_embeds_list, axis=1),
                tf.concat(posit_embeds_list, axis=1),
                tf.concat(token_embeds_list, axis=1),
                self.num_layer,
                self.size,
                masks=attn_masks,
                dropout=self.dropout,
                is_training=self.is_training,
                scope="transformer")
            outputs = outputs[:,end_idx:,-1]
            outputs = fully_connected(
                outputs,
                self.size,
                dropout=self.dropout,
                is_training=self.is_training,
                scope="projs")
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
                          tf.cast(symbol, tf.int32))
        score *= tf.cast(tf.logical_not(mask), tf.float32)
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
            tf.cast(tf.map_fn(fn,
                                  pths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  back_prop=False,
                                  swap_memory=True), tf.float32),
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
            tf.cast(tf.map_fn(fn,
                                  pths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  back_prop=False,
                                  swap_memory=True), tf.float32),
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
    indices = tf.cast(tf.multinomial(scores * (7**gamma), num_candidates), tf.int32)
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
    padded_seq_masks = tf.not_equal(padded_seqs, 0)
    padded_seg_masks1 = tf.logical_or(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_and(padded_segs, padded_seg_masks1)
    padded_seg_masks2 = tf.logical_xor(padded_seq_masks[:,:-1], padded_seq_masks[:,1:])
    padded_segs = tf.logical_or(padded_segs, padded_seg_masks2)

    num_words = tf.reduce_sum(tf.cast(padded_segs, tf.int32), axis=1)-1
    max_num_word = tf.maximum(tf.reduce_max(num_words), 1)

    def get_idx(padded_seg):
        idx = tf.range(max_length+1, dtype=tf.int32)
        idx = tf.boolean_mask(idx, padded_seg)
        num = tf.shape(idx)[0]-1
        start = tf.pad(idx[:-1], [[0,max_num_word-num]])
        start = tf.reshape(start, [max_num_word])
        length = tf.pad(idx[1:] - idx[:-1], [[0,max_num_word-num]])
        length = tf.reshape(length, [max_num_word])
        return start, length

    starts, lengths = tf.map_fn(
        get_idx,
        padded_segs,
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
        return seq

    seqs = tf.map_fn(
        stitch,
        segmented_seqs,
        tf.int32,
        parallel_iterations=128,
        back_prop=False,
        swap_memory=True)
    return seqs

def unique_2d(x):
    """
    x: num_vectors x vector_dim
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
