# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Talking Head Attention layer."""
# pylint: disable=g-classes-have-attributes
import math
import string
import gin
import tensorflow as tf
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import function

import tensorflow_probability as tfp
import tf_utils

_CHR_IDX = string.ascii_lowercase
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax


    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)
    
def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              cache=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Memory-efficient Multihead scaled-dot-product self-attention.

  Based on Tensor2Tensor version but adds optional caching.

  Returns multihead-self-attention(layer_norm(x))

  Computes one attention head at a time to avoid exhausting memory.

  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.

  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    cache: Optional dict containing tensors which are the results of previous
        attentions, used for fast decoding. Expects the dict to contain two
        keys ('k' and 'v'), for the initial call the values for these keys
        should be empty Tensors of the appropriate shape.
        'k' [batch_size, 0, key_channels] 'v' [batch_size, 0, value_channels]
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads
  num_batch_dims = qkv_rank - len(self._attention_axes) - 2
    # print("Using Talking Heads Attention")
    # # The shape of attn_scores is:
    # # (<batch_dims>, num_heads, <query_attn_dims>, <key_attn_dims>)
    # attn_scores_rank = num_batch_dims + 1 + len(self._attention_axes) * 2
    # scores_notation = _CHR_IDX[:attn_scores_rank]
    # projection_notation = scores_notation[num_batch_dims] + (
    #     _CHR_IDX[attn_scores_rank])
    # projected_scores_notation = scores_notation[:num_batch_dims] + (
    #     _CHR_IDX[attn_scores_rank] + scores_notation[num_batch_dims + 1:])
    # self._talking_heads_equation = "%s,%s->%s" % (
    #     scores_notation, projection_notation, projected_scores_notation)

    # self._pre_softmax_weight = self.add_weight(
    #     "pre_softmax_weight",
    #     shape=(self._num_heads, self._num_heads),
    #     initializer=tf_utils.clone_initializer(self._kernel_initializer),
    #     regularizer=self._kernel_regularizer,
    #     constraint=self._kernel_constraint,
    #     dtype=self.dtype,
    #     trainable=True)
    # self._post_softmax_weight = self.add_weight(
    #     "post_softmax_weight",
    #     shape=(self._num_heads, self._num_heads),
    #     initializer=tf_utils.clone_initializer(self._kernel_initializer),
    #     regularizer=self._kernel_regularizer,
    #     constraint=self._kernel_constraint,
    #     dtype=self.dtype,
    #     trainable=True)
  def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    if cache is not None:
      cache_k = []
      cache_v = []
    for h in range(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, 'SAME')
        q, k, v = tf.split(combined, 3, axis=2)
        if cache is not None:
          k = tf.concat([cache['k'][:, h], k], axis=1)
          v = tf.concat([cache['v'][:, h], v], axis=1)
          cache_k.append(k)
          cache_v.append(v)
        o = common_attention.scaled_dot_product_attention_simple(
            q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, 'SAME')
    if cache is not None:
      cache['k'] = tf.stack(cache_k, axis=1)
      cache['v'] = tf.stack(cache_v, axis=1)
    return y

  
  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_name='multihead_attention', values=[x]):
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          'wqkv', [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          'wo', [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())  #  pytype: disable=attribute-error
    return y
  
# @tf.keras.utils.register_keras_serializable(package="Text")
# @gin.configurable
# class TalkingHeadsAttention(tf.keras.layers.MultiHeadAttention):
#   """Implements Talking-Heads Attention.

#   This is an implementation of Talking-Heads Attention based on the paper
#   Talking-Heads Attention (https://arxiv.org/abs/2003.02436): it enhanced
#   multi-head attention by including linearprojections across the attention-heads
#   dimension, immediately before and after the softmax operation.

#   See the base class `tf.keras.layers.MultiHeadAttention` for more details.

#   Args:
#     num_heads: Number of attention heads.
#     key_dim: Size of each attention head for query and key.
#     value_dim:  Size of each attention head for value.
#     dropout: Dropout probability.
#     use_bias: Boolean, whether the dense layers use bias vectors/matrices.
#     output_shape: The expected shape of an output tensor, besides the batch and
#       sequence dims. If not specified, projects back to the key feature dim.
#     attention_axes: axes over which the attention is applied. `None` means
#       attention over all axes, but batch, heads, and features.
#     return_attention_scores: bool, if `True`, returns the multi-head attention
#       scores as an additional output argument.
#     kernel_initializer: Initializer for dense layer kernels.
#     bias_initializer: Initializer for dense layer biases.
#     kernel_regularizer: Regularizer for dense layer kernels.
#     bias_regularizer: Regularizer for dense layer biases.
#     activity_regularizer: Regularizer for dense layer activity.
#     kernel_constraint: Constraint for dense layer kernels.
#     bias_constraint: Constraint for dense layer kernels.
#   """

#   def _build_attention(self, qkv_rank):
#     """Builds multi-head dot-product attention computations.

#     This function overrides base class to create additional linear projection
#     that will be applied on attention scores before and after softmax.

#     Args:
#       qkv_rank: The rank of query, key, value tensors after projection.
#     """
#     super(TalkingHeadsAttention, self)._build_attention(qkv_rank)

#     # Build an equation:
#     # (<batch_dims>, num_heads_a, ...),(num_heads_a, num_heads_b) ->
#     # (<batch_dims>, num_heads_b, ...)
#     # qkv_ranks has `batch_dims`, `attention_dims`, `num_heads` and `channels`.
#     num_batch_dims = qkv_rank - len(self._attention_axes) - 2
#     print("Using Talking Heads Attention")
#     # The shape of attn_scores is:
#     # (<batch_dims>, num_heads, <query_attn_dims>, <key_attn_dims>)
#     attn_scores_rank = num_batch_dims + 1 + len(self._attention_axes) * 2
#     scores_notation = _CHR_IDX[:attn_scores_rank]
#     projection_notation = scores_notation[num_batch_dims] + (
#         _CHR_IDX[attn_scores_rank])
#     projected_scores_notation = scores_notation[:num_batch_dims] + (
#         _CHR_IDX[attn_scores_rank] + scores_notation[num_batch_dims + 1:])
#     self._talking_heads_equation = "%s,%s->%s" % (
#         scores_notation, projection_notation, projected_scores_notation)

#     self._pre_softmax_weight = self.add_weight(
#         "pre_softmax_weight",
#         shape=(self._num_heads, self._num_heads),
#         initializer=tf_utils.clone_initializer(self._kernel_initializer),
#         regularizer=self._kernel_regularizer,
#         constraint=self._kernel_constraint,
#         dtype=self.dtype,
#         trainable=True)
#     self._post_softmax_weight = self.add_weight(
#         "post_softmax_weight",
#         shape=(self._num_heads, self._num_heads),
#         initializer=tf_utils.clone_initializer(self._kernel_initializer),
#         regularizer=self._kernel_regularizer,
#         constraint=self._kernel_constraint,
#         dtype=self.dtype,
#         trainable=True)

#   def _compute_attention(self,
#                          query_tensor,
#                          key_tensor,
#                          value_tensor,
#                          attention_mask=None,
#                          training=None):
#     """Applies Dot-product attention with query, key, value tensors.

#     This function overrides base class to apply additional linear projection
#     on attention scores before and after softmax.

#     Args:
#       query_tensor: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
#       key_tensor: Projected key `Tensor` of shape `[B, T, N, key_dim]`.
#       value_tensor: Projected value `Tensor` of shape `[B, T, N, value_dim]`.
#       attention_mask: a boolean mask of shape `[B, T, S]`, that prevents
#         attention to certain positions.
#       training: Python boolean indicating whether the layer should behave in
#         training mode (adding dropout) or in inference mode (doing nothing).

#     Returns:
#       attention_output: Multi-headed outputs of attention computation.
#       attention_scores: Multi-headed attention weights.
#     """
#     # Take the dot product between "query" and "key" to get the raw
#     # attention scores.
#     attention_scores = tf.einsum(self._dot_product_equation, key_tensor,
#                                  query_tensor)
#     attention_scores = tf.multiply(attention_scores,
#                                    1.0 / math.sqrt(float(self._key_dim)))

#     # Apply linear projection before softmax
#     attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
#                                  self._pre_softmax_weight)

#     # Normalize the attention scores to probabilities.
#     # `attention_scores` = [B, N, T, S]
#     attention_scores = self._masked_softmax(attention_scores, attention_mask)

#     # Apply linear projection after softmax
#     attention_scores = tf.einsum(self._talking_heads_equation, attention_scores,
#                                  self._post_softmax_weight)

#     # This is actually dropping out entire tokens to attend to, which might
#     # seem a bit unusual, but is taken from the original Transformer paper.
#     attention_scores_dropout = self._dropout_layer(
#         attention_scores, training=training)

#     # `context_layer` = [B, T, N, H]
#     attention_output = tf.einsum(self._combine_equation,
#                                  attention_scores_dropout, value_tensor)
#     return attention_output, attention_scores
