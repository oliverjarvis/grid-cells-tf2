# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model for grid cells supervised training.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
from tensorflow.keras import layers

def displaced_linear_initializer(input_size, displace):
  stddev = 1. / numpy.sqrt(input_size)
  return tf.keras.initializers.TruncatedNormal(mean=displace*stddev, stddev=stddev)

class MinimalRNNCell(layers.Layer):
    def __init__(self,
               target_ensembles,
               nh_lstm,
               nh_bottleneck,
               nh_embed=None,
               dropoutrates_bottleneck=None,
               bottleneck_weight_decay=0.0,
               bottleneck_has_bias=False,
               init_weight_disp=0.0, **kwargs):
      super(MinimalRNNCell, self).__init__(**kwargs)
      self._target_ensembles = target_ensembles
      self._nh_embed = nh_embed
      self._nh_lstm = nh_lstm
      self._nh_bottleneck = nh_bottleneck
      self._dropoutrates_bottleneck = dropoutrates_bottleneck
      self._bottleneck_weight_decay = bottleneck_weight_decay
      self._bottleneck_has_bias = bottleneck_has_bias
      self._init_weight_disp = init_weight_disp
      self.training = False
      self.state_size = (nh_lstm, nh_lstm)
      #defining layers
      self.lstm = layers.LSTMCell(units=self._nh_lstm, name="trash")
      #self.lstm = layers.LSTM(self._nh_lstm, return_state=True)
      self.bottleneck = layers.Dense(self._nh_bottleneck, name="bottleneck",
                                     use_bias=self._bottleneck_has_bias,
                                      kernel_regularizer=tf.keras.regularizers.L2(self._bottleneck_weight_decay))
      #ISSUE IS THIS https://stackoverflow.com/questions/52671481/why-are-variables-defined-with-self-automatically-given-a-listwrapper-while
      self.output_layers = []
      for ens in self._target_ensembles:
        dense = layers.Dense(units=ens.n_cells, name="pc_logits", kernel_regularizer = tf.keras.regularizers.L2(self._bottleneck_weight_decay), kernel_initializer = displaced_linear_initializer(self._nh_bottleneck,self._init_weight_disp))
        self.output_layers.append(dense)
      
      self.dropout = layers.Dropout(self._dropoutrates_bottleneck, name="dropout")
      #self.output_layers = [
      #for ens in self._target_ensembles]
      #self.output_layers = list(self.output_layers)

    def call(self, inputs, states):
      conc_inputs = tf.concat(inputs, axis=1)
      lstm_inputs = conc_inputs
      lstm_output = self.lstm(lstm_inputs, states=states)
      next_state = lstm_output[1:][0]
      lstm_output = lstm_output[0]
      bottleneck = self.bottleneck(lstm_output)
      if self.training:
        bottleneck = self.dropout(bottleneck)
      ens_outputs = [
        layer(bottleneck)
      for layer in self.output_layers]
      return (ens_outputs, bottleneck, lstm_output), tuple(list(next_state))

class GridCellNetwork(tf.keras.models.Model):
  def __init__(
    self,
    target_ensembles,
    nh_lstm,
    nh_bottleneck,
    dropoutrates_bottleneck,
    bottleneck_weight_decay,
    bottleneck_has_bias,
    init_weight_disp,
    **kwargs):
    super(GridCellNetwork, self).__init__(**kwargs)

    self._target_ensembles = target_ensembles
    self._nh_lstm = nh_lstm
    self._nh_bottleneck = nh_bottleneck
    self._dropoutrates_botleneck = dropoutrates_bottleneck
    self._bottleneck_weight_decay = bottleneck_weight_decay
    self._bottleneck_has_bias = bottleneck_has_bias
    self._init_weight_disp = bottleneck_has_bias

    self.init_lstm_state = layers.Dense(self._nh_lstm, name="state_init")
    self.init_lstm_cell = layers.Dense(self._nh_lstm, name="cell_init") 
    self.rnn_core = MinimalRNNCell(
      target_ensembles = target_ensembles,
      nh_lstm = nh_lstm,
      nh_bottleneck = nh_bottleneck,
      dropoutrates_bottleneck=dropoutrates_bottleneck,
      bottleneck_weight_decay=bottleneck_weight_decay,
      bottleneck_has_bias=bottleneck_has_bias,
      init_weight_disp=init_weight_disp
    )
    self.RNN = layers.RNN(return_state=True, return_sequences=True, cell=self.rnn_core)

  def call(self, velocities, initial_conditions, trainable=False):
    concat_initial = tf.concat(initial_conditions, axis=1)
    init_lstm_state = self.init_lstm_state(concat_initial)
    init_lstm_cell = self.init_lstm_cell(concat_initial)    
    output_seq = self.RNN((velocities,), initial_state=(init_lstm_state, init_lstm_cell))
    final_state = output_seq[-2:]

    output_seq = output_seq[0]

    ens_targets = output_seq[0]
    bottleneck = output_seq[1]
    lstm_output = output_seq[2]
    return (ens_targets, bottleneck, lstm_output), final_state