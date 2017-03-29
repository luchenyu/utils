
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


### Building Blocks ###

# fully_connected layer
def fully_connected(inputs,
                    num_outputs, 
                    decay=0.99999, 
                    activation_fn=None, 
                    is_training=True, 
                    reuse=None, 
                    scope=None):
  """Adds a fully connected layer.

  """
  if not isinstance(num_outputs, int):
    raise ValueError('num_outputs should be integer, got %s.', num_outputs)
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
    weights = tf.contrib.framework.model_variable('weights', 
                                                  shape=weights_shape, 
                                                  dtype=dtype, 
                                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                                  collections=tf.GraphKeys.WEIGHTS, 
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable('biases', 
                                                 shape=[num_outputs,], 
                                                 dtype=dtype, 
                                                 initializer=tf.zeros_initializer(), 
                                                 collections=tf.GraphKeys.BIASES, 
                                                 trainable=True)
    if len(static_shape) > 2:
      # Reshape inputs
      inputs = tf.reshape(inputs, [-1, num_input_units])
    outputs = tf.matmul(inputs, weights)
    moving_mean = tf.contrib.framework.model_variable('moving_mean', 
                                                      shape=[num_outputs,], 
                                                      dtype=dtype, 
                                                      initializer=tf.zeros_initializer(), 
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if activation_fn:
      outputs = activation_fn(outputs)
    if len(static_shape) > 2:
      # Reshape back outputs
      outputs = tf.reshape(outputs, tf.stack(out_shape))
      outputs.set_shape(static_shape)
    return outputs

# convolutional layer
def convolution2d(inputs, 
                  num_outputs, 
                  kernel_size, 
                  pool_size=None, 
                  decay=0.99999, 
                  activation_fn=None, 
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
    weights_shape = list(kernel_size) + [num_filters_in, num_outputs]
    weights = tf.contrib.framework.model_variable(name='weights', 
                                                  shape=weights_shape, 
                                                  dtype=dtype,
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  collections=tf.GraphKeys.WEIGHTS,
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable(name='biases',
                                                 shape=[num_outputs,],
                                                 dtype=dtype,
                                                 initializer=tf.zeros_initializer(),
                                                 collections=tf.GraphKeys.BIASES,
                                                 trainable=True)
    outputs = tf.nn.conv2d(inputs, weights, [1,1,1,1], padding='SAME')
    moving_mean = tf.contrib.framework.model_variable('moving_mean',
                                                      shape=[num_outputs,],
                                                      dtype=dtype,
                                                      initializer=tf.zeros_initializer(),
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0, 1, 2], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
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
  params = tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)
  while len(params) > 0:
    p = params.pop()
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, 
        p.assign(decay*p + (1-decay)*tf.truncated_normal(p.get_shape(), stddev=0.01)))


### Nets ###

# ResDNN
def ResDNN(inputs, 
           out_size, 
           num_layers, 
           decay=0.99999, 
           activation_fn=tf.nn.relu, 
           is_training=True, 
           reuse=None, 
           scope=None):
  """ a deep neural net with fully connected layers

  """
  with tf.variable_scope(scope,
                            "ResDNN",
                            [inputs],
                            reuse=reuse):
    # first layer
    with tf.variable_scope("layer{0}".format(0)):
      outputs = fully_connected(inputs, out_size, decay=decay, activation_fn=None, is_training=is_training)
    # residual layers
    for i in xrange(num_layers-1):
      inputs = activation_fn(outputs)
      with tf.variable_scope("layer{0}".format(i+1)):
        outputs += fully_connected(inputs, out_size, decay=decay, activation_fn=None, is_training=is_training)
    if is_training:
      outputs = tf.nn.dropout(outputs, 0.8)
    return outputs

# ResCNN
def ResCNN(inputs, 
           out_size, 
           num_layers, 
           kernel_size, 
           pool_size, 
           decay=0.99999, 
           activation_fn=tf.nn.relu, 
           is_training=True, 
           reuse=None, 
           scope=None):
  """ a convolutaional neural net with conv2d and max_pool layers

  """
  with tf.variable_scope(scope,
                         "ResCNN",
                         [inputs],
                         reuse=reuse):
    # first layer
    with tf.variable_scope("layer{0}".format(0)):
      outputs = convolution2d(inputs, out_size, kernel_size, decay=decay, activation_fn=None, 
          is_training=is_training)
    # residual layers
    for i in xrange(num_layers-1):
      with tf.variable_scope("layer{0}".format(i+1)):
        if pool_size:
          pool_shape = [1] + list(pool_size) + [1]
          outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
        inputs = activation_fn(outputs)
        outputs += convolution2d(inputs, out_size, kernel_size, decay=decay, activation_fn=None, 
            is_training=is_training)
    if is_training:
      outputs = tf.nn.dropout(outputs, 0.8)
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
        r, u = tf.split(1, 2, self._linear(tf.concat([inputs, state], 1),
                                             2 * self._num_units))
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = self._activation(self._linear(tf.concat([inputs, r * state], 1),
                                     self._num_units))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class newGRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.nn.relu, linear=fully_connected):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
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
    with tf.variable_scope(scope or "newGRUCell",
                           None,
                           [inputs, state],
                           reuse=None):  # "GRUCell"
      with tf.variable_scope("InputGates"):
        i = self._linear(state, self._num_units)
        i = tf.sigmoid(i)
      with tf.variable_scope("ForgetGatesAndUpdates"):
        x = self._linear(inputs, 2 * self._num_units)
        c, f = tf.split(x, 2, 1)
        c = self._activation(c)
        f = tf.sigmoid(f)
        update = state + i * c
        new_h = f * update
    return update - new_h, new_h

class LayerNormFastWeightsBasicRNNCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, forget_bias=1.0, reuse_norm=False,
               input_size=None, activation=tf.nn.relu,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               loop_steps=1, decay_rate=0.9, learning_rate=0.5,
               dropout_keep_prob=1.0, dropout_prob_seed=None):

    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._reuse_norm = reuse_norm
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._S = loop_steps
    self._eta = learning_rate
    self._lambda = decay_rate
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope=None):
    reuse = tf.get_variable_scope().reuse
    with tf.variable_scope(scope or "Norm") as scope:
      normalized = tf.contrib.layers.layer_norm(inp, reuse=reuse, scope=scope)
      return normalized

  def _fwlinear(self, args, output_size, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
    assert len(args) == 2
    assert args[0].get_shape().as_list()[1] == output_size

    dtype = [a.dtype for a in args][0]

    with tf.variable_scope(scope or "Linear"):
      matrixW = tf.get_variable(
        "MatrixW", dtype=dtype, initializer=tf.convert_to_tensor(np.eye(output_size, dtype=np.float32) * .05))

      matrixC = tf.get_variable(
        "MatrixC", [args[1].get_shape().as_list()[1], output_size], dtype=dtype)

      res = tf.matmul(args[0], matrixW) + tf.matmul(args[1], matrixC)
      return res

  def zero_fast_weights(self, batch_size, dtype):
    """Return zero-filled fast_weights tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      A zero filled fast_weights of shape [batch_size, state_size, state_size]
    """
    state_size = self.state_size

    zeros = tf.zeros(
        tf.stack([batch_size, state_size, state_size]), dtype=dtype)
    zeros.set_shape([None, state_size, state_size])

    return zeros

  def _vector2matrix(self, vector):
    memory_size = vector.get_shape().as_list()[1]
    return tf.reshape(vector, [-1, memory_size, 1])

  def _matrix2vector(self, matrix):
    return tf.squeeze(matrix, [2])

  def __call__(self, inputs, state, scope=None):
    fast_weights = tf.slice(state, [0, 0], [-1, self._num_units*self._num_units])
    fast_weights = tf.reshape(fast_weights, [-1, self._num_units, self._num_units])
    state = tf.slice(state, [0, self._num_units*self._num_units], [-1, self._num_units])
    with tf.variable_scope(scope or type(self).__name__) as scope:
      """Compute Wh(t) + Cx(t)"""
      linear = self._fwlinear([state, inputs], self._num_units, False)
      """Compute h_0(t+1) = f(Wh(t) + Cx(t))"""
      if not self._reuse_norm:
        h = self._activation(self._norm(linear, scope="Norm0"))
      else:
        h = self._activation(self._norm(linear))
      h = self._vector2matrix(h)
      linear = self._vector2matrix(linear)
      for i in range(self._S):
        """
        Compute h_{s+1}(t+1) = f([Wh(t) + Cx(t)] + A(t) h_s(t+1)), S times.
        See Eqn (2) in the paper.
        """
        if not self._reuse_norm:
          h = self._activation(self._norm(linear +
                                          tf.matmul(fast_weights, h), scope="Norm%d" % (i + 1)))
        else:
          h = self._activation(self._norm(linear +
                                          tf.matmul(fast_weights, h)))

      """
      Compute A(t+1)  according to Eqn (4)
      """
      state = self._vector2matrix(state)
      new_fast_weights = self._lambda * fast_weights + self._eta * tf.matmul(state, state, adj_y=True)

      h = self._matrix2vector(h)
      new_fast_weights = tf.reshape(new_fast_weights, [-1, self._num_units*self._num_units])

      return h, tf.concat([new_fast_weights, h], 1)

def create_cell(size, num_layers, cell_type="GRU", decay=0.99999, is_training=True):
  # fully connected layers inside the rnn cell
  def _linear(inputs, num_outputs):
    return fully_connected(inputs, num_outputs, decay=decay, is_training=is_training) 
 
  # build single cell
  if cell_type == "GRU":
    single_cell = newGRUCell(size, activation=tf.tanh, linear=_linear)
  elif cell_type == "LSTM":
    single_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True, cell_clip=5.0, num_proj=size)
  elif cell_type == "FastWeight":
    single_cell = LayerNormFastWeightsBasicRNNCell(size)
  else:
    raise ValueError('Incorrect cell type! (GRU|LSTM)')
  cell = single_cell
  # stack multiple cells
  if num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
  if is_training:
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
  return cell


### Recurrent Decoders ###

# greedy decoder
def greedy_dec(length,
               initial_state,
               input_embedding,
               output_embedding,
               iter_fn,
               logit_fn=None,
               memory=None):
  """ A greedy decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = input_embedding.get_shape()[1].value
  inputs = tf.nn.embedding_lookup(input_embedding, tf.zeros([batch_size], dtype=tf.int32))

  outputs, state = iter_fn(inputs, initial_state, memory) if memory else iter_fn(inputs, initial_state)
  logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

  symbol = tf.argmax(logits, 1)
  seq = [symbol]
  mask = tf.not_equal(symbol, 0)
  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    inputs = tf.nn.embedding_lookup(input_embedding, symbol)

    outputs, state = iter_fn(inputs, state, memory) if memory else iter_fn(inputs, initial_state)
    logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

    symbol = tf.argmax(logits, 1)
    symbol = tf.where(mask, symbol, tf.zeros([batch_size], dtype=tf.int64))
    mask = tf.not_equal(symbol, 0)

    seq.append(symbol)

  return tf.expand_dims(tf.stack(seq, 1), 1)

def stochastic_dec(length,
                   initial_state,
                   input_embedding,
                   output_embedding,
                   iter_fn,
                   logit_fn=None,
                   memory=None,
                   num_candidates=1):
  """ A stochastic decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = input_embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])

  outputs, state = iter_fn(inputs, initial_state, memory) if memory else iter_fn(inputs, initial_state)
  logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

  symbol = tf.reshape(tf.multinomial(logits, num_candidates), [-1])
  mask = tf.equal(symbol, 0)
  seq = [symbol]
  tf.get_variable_scope().reuse_variables()
  if isinstance(state, tuple):
    state = tuple([tf.reshape(tf.stack([s]*num_candidates, axis=1), 
        [batch_size*num_candidates, s.get_shape()[1].value]) for s in state])
  else:
    state = tf.reshape(tf.stack([state]*num_candidates, axis=1), 
        [batch_size*num_candidates, state.get_shape()[1].value])

  for _ in xrange(length-1):

    inputs = tf.nn.embedding_lookup(input_embedding, symbol)

    outputs, state = iter_fn(inputs, state, memory) if memory else iter_fn(inputs, initial_state)
    logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

    symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
    symbol = tf.where(mask, tf.to_int64(tf.zeros([batch_size*num_candidates])), symbol)
    mask = tf.equal(symbol, 0)
    seq.append(symbol)

  return tf.reshape(tf.stack(seq, 1), [batch_size, num_candidates, length])

# beam decoder
def beam_dec(length,
             initial_state,
             input_embedding,
             output_embedding,
             iter_fn,
             logit_fn=None,
             memory=None,
             num_candidates=1,
             beam_size=100,
             gamma=0.65):
  """ A basic beam decoder

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = input_embedding.get_shape()[1].value
  inputs = tf.nn.embedding_lookup(input_embedding, tf.zeros([batch_size], dtype=tf.int32))
  vocab_size = tf.shape(input_embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, memory) if memory else iter_fn(inputs, initial_state)
  logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

  prev = tf.nn.log_softmax(logits)
  probs = tf.slice(prev, [0, 1], [-1, -1])
  best_probs, indices = tf.nn.top_k(probs, beam_size)

  symbols = indices % vocab_size + 1
  beam_parent = indices // vocab_size
  beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
  paths = tf.reshape(symbols, [-1, 1])

  if isinstance(memory, tuple):
    memory_prime = []
    for m in memory:
      mdim = [d.value for d in m.get_shape()]
      m = tf.expand_dims(m, 1)
      memory_prime.append(tf.reshape(tf.tile(m, [1] + [beam_size] + [1]*(len(mdim)-1)),
          [-1]+mdim[1:]))
    memory = tuple(memory_prime)
  elif memory != None:
    mdim = [d.value for d in memory.get_shape()]
    memory = tf.expand_dims(memory, 1)
    memory = tf.reshape(tf.tile(memory, [1] + [beam_size] + [1]*(len(mdim)-1)),
        [-1]+mdim[1:])

  candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
  scores = [tf.slice(prev, [0, 0], [-1, 1])]

  tf.get_variable_scope().reuse_variables()
  for i in xrange(length-1):

    if isinstance(state, tuple):
      state = tuple([tf.gather(s, beam_parent) for s in state])
    else:
      state = tf.gather(state, beam_parent)

    inputs = tf.reshape(tf.nn.embedding_lookup(input_embedding, symbols), [-1, inputs_size])

    # iter
    outputs, state = iter_fn(inputs, state, memory) if memory else iter_fn(inputs, initial_state)
    logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

    prev = tf.reshape(tf.nn.log_softmax(logits), [batch_size, beam_size, vocab_size])
    prev += tf.expand_dims(best_probs, 2)

    # add the path and score of the candidates in the current beam to the lists
    close_score = tf.squeeze(tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2]) / (float(i+2) ** gamma)
    candidates.append(tf.reshape(tf.pad(paths, [[0, 0], [0, length-1-i]], "CONSTANT"), 
        [batch_size, beam_size, length]))
    scores.append(close_score)

    probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % (vocab_size - 1) + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent, [-1])
    paths = tf.gather(paths, beam_parent)
    paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

  # pick the topk from the candidates in the lists
  candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
  scores = tf.concat(scores, 1)
  best_scores, indices = tf.nn.top_k(scores, num_candidates)
  indices = tf.reshape(tf.expand_dims(tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices, [-1])
  best_candidates = tf.reshape(tf.gather(candidates, indices), [batch_size, num_candidates, length])

  return best_candidates, best_scores

# beam decoder
def stochastic_beam_dec(length,
                        initial_state,
                        input_embedding,
                        output_embedding,
                        iter_fn,
                        logit_fn=None,
                        memory=None,
                        num_candidates=1,
                        beam_size=100,
                        gamma=0.65):
  """ A stochastic beam decoder

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = input_embedding.get_shape()[1].value
  inputs = tf.nn.embedding_lookup(input_embedding, tf.zeros([batch_size], dtype=tf.int32))
  vocab_size = tf.shape(input_embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, memory) if memory else iter_fn(inputs, initial_state)
  logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

  prev = tf.nn.log_softmax(logits)
  probs = tf.slice(prev, [0, 1], [-1, -1])
  best_probs, indices = tf.nn.top_k(probs, beam_size)

  symbols = indices % vocab_size + 1
  beam_parent = indices // vocab_size
  beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
  paths = tf.reshape(symbols, [-1, 1])

  if isinstance(memory, tuple):
    memory_prime = []
    for m in memory:
      mdim = [d.value for d in m.get_shape()]
      m = tf.expand_dims(m, 1)
      memory_prime.append(tf.reshape(tf.tile(m, [1] + [beam_size] + [1]*(len(mdim)-1)),
          [-1]+mdim[1:]))
    memory = tuple(memory_prime)
  elif memory != None:
    mdim = [d.value for d in memory.get_shape()]
    memory = tf.expand_dims(memory, 1)
    memory = tf.reshape(tf.tile(memory, [1] + [beam_size] + [1]*(len(mdim)-1)),
        [-1]+mdim[1:])

  candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
  scores = [tf.slice(prev, [0, 0], [-1, 1])]

  tf.get_variable_scope().reuse_variables()
  for i in xrange(length-1):

    if isinstance(state, tuple):
      state = tuple([tf.gather(s, beam_parent) for s in state])
    else:
      state = tf.gather(state, beam_parent)

    inputs = tf.reshape(tf.nn.embedding_lookup(input_embedding, symbols), [-1, inputs_size])

    # iter
    outputs, state = iter_fn(inputs, state, memory) if memory else iter_fn(inputs, initial_state)
    logits = logit_fn(outputs) if logit_fn else tf.matmul(outputs, tf.transpose(output_embedding))

    prev = tf.reshape(tf.nn.log_softmax(logits), [batch_size, beam_size, vocab_size])
    prev += tf.expand_dims(best_probs, 2)

    # add the path and score of the candidates in the current beam to the lists
    close_score = tf.squeeze(tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2]) / (float(i+2) ** gamma)
    candidates.append(tf.reshape(tf.pad(paths, [[0, 0], [0, length-1-i]], "CONSTANT"), 
        [batch_size, beam_size, length]))
    scores.append(close_score)

    probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % (vocab_size - 1) + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent, [-1])
    paths = tf.gather(paths, beam_parent)
    paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

  # pick the topk from the candidates in the lists
  candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
  scores = tf.concat(scores, 1)
  indices = tf.to_int32(tf.multinomial(scores*5, num_candidates))
  indices = tf.reshape(tf.expand_dims(tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices, [-1])
  best_candidates = tf.reshape(tf.gather(candidates, indices), [batch_size, num_candidates, length])
  best_scores = tf.reshape(tf.gather(tf.reshape(scores, [-1]), indices), [batch_size, num_candidates])

  return best_candidates, best_scores

### Attention on Memory ###

def attention(query,
              keys,
              values,
              patch=None,
              is_training=True):
  """ implements the attention mechanism

  query: [batch_size x dim]
  keys: [batch_size x length x dim]
  values: [batch_size x length x dim]
  """
  query = tf.expand_dims(query, 1)
  logits = convolution2d(tf.expand_dims(tf.tanh(query+keys), 1), 1, [1, 1], 
      is_training=is_training, scope="attention")
  logits = tf.squeeze(logits, [1, 3]) + patch if patch != None else tf.squeeze(logits, [1, 3])
  results = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(logits), 2) * values, [1])
  return results

def attention_iter(inputs,
                   state,
                   memory,
                   cell,
                   is_training=True,
                   reuse=None):
  """ implements an attention iter function

  """
  if len(memory) == 2:
    keys, values = memory
    patch = None
  elif len(memory) == 3:
    keys, values, patch = memory
  if not values.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % values.get_shape())

  attn_feat = state[0]
  cell_state = state[1:] if len(state) > 2 else state[1]
  batch_size = tf.shape(inputs)[0]
  size = inputs.get_shape()[1].value
  mem_size = values.get_shape()[2].value

  with tf.variable_scope("attention_decoder", reuse=reuse):

    with tf.variable_scope("dec_cell"):
      cell_outputs, cell_state = cell(tf.concat([inputs, attn_feat], 1), cell_state)

    with tf.variable_scope("query"):
      query = fully_connected(cell_outputs, mem_size, 
          activation_fn=None, is_training=is_training)

    with tf.variable_scope("attention"):
      attn_feat = attention(query, keys, values, patch, is_training)

    with tf.variable_scope("output_proj"):
      outputs = ResDNN(tf.concat([cell_outputs, attn_feat], 1), size, 2, 
          is_training=is_training)

    cell_state = cell_state if isinstance(cell_state, (tuple, list)) else (cell_state,)
    state = (attn_feat,) + cell_state

  return outputs, state

### Copy Mechanism ###

def make_logit_fn(char_embedding, source_embedding, source_ids, is_training=True):
  def logit_fn(outputs):
    batch_size = tf.shape(source_embedding)[0]
    length = source_embedding.get_shape()[1].value
    size = outputs.get_shape()[-1].value
    vocab_size = char_embedding.get_shape()[0].value
    if outputs.get_shape().ndims == 3:
      beam_size = outputs.get_shape()[1].value
      logits_static = tf.reshape(tf.matmul(tf.reshape(outputs, [-1, size]), tf.transpose(char_embedding)), 
          [batch_size, beam_size, vocab_size])
      logits_ptr = tf.matmul(outputs, tf.transpose(source_embedding, [0, 2, 1]))
    else:
      assert(outputs.get_shape().ndims == 2)
      logits_static = tf.reshape(tf.matmul(outputs, tf.transpose(char_embedding)), [batch_size, -1, vocab_size])
      logits_ptr = tf.matmul(tf.reshape(outputs, [batch_size, -1, size]),
          tf.transpose(source_embedding, [0, 2, 1]))
      beam_size = tf.shape(logits_ptr)[1]
    data = tf.reshape(logits_ptr, [-1])
    indices = tf.reshape(tf.reshape(tf.tile(tf.expand_dims(source_ids, 1), [1, beam_size, 1]),
        [-1, length]) + tf.expand_dims(tf.range(batch_size*beam_size) * vocab_size, 1), [-1])
    logits_src = tf.reshape(tf.unsorted_segment_sum(data, indices, batch_size*beam_size*vocab_size),
        [batch_size, beam_size, vocab_size])
    logits_src = tf.concat([tf.zeros([batch_size, beam_size, 1]), tf.slice(logits_src, [0, 0, 1], [-1, -1, -1])], 2)
    logits = logits_static + logits_src
    if outputs.get_shape().ndims == 2:
      logits = tf.reshape(logits, [-1, vocab_size])
    return logits
  return logit_fn

