# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Image classification Model.

This model works on images converted to embeddings
using the bottleneck layer of pre-trained image classification
model.

Actual model is a simple 2 layer network with 1 hidden layer.
Input can be any set of labeled images preprocessed by
//path/to/prepare_images.
"""

import math

import google.cloud.ml.models as models
import tensorflow as tf


class ImageClassification(models.Classification):
  """Class for creating an image classification network."""

  def __init__(self):
    super(ImageClassification, self).__init__()

  def create_graph(self):
    with tf.Graph().as_default() as g:
      self._features = {
          'image_uri': tf.FixedLenFeature(shape=[],
                                          dtype=tf.string,
                                          default_value=['']),
          # TODO(b/27362640): Add support for multi-label.
          'label': tf.FixedLenFeature(shape=[1],
                                      dtype=tf.int64,
                                      default_value=[-1]),
          'embedding': tf.FixedLenFeature(
              shape=[self.hyperparams.embedding_size],
              dtype=tf.float32)
      }

      self._create_inputs()
      train, prediction = self._create_hidden_layers()
      self._create_outputs(prediction)
      self._create_training(train)
      self._create_initialization()
    return g

  def _create_inputs(self):

    def encode_targets(targets, batch_size, labels):
      """Convert our target tensor to one-of-k ("one-hot") encoding.

      For each row, the output will have batch_size elements, with batch_size
      equal to the number of possible labels, and each row will have only
      one value set to 1, corresponding to the target label, with
      all other elements being 0.
      Args:
        targets: list of target label values
        batch_size: number of targets in the batch
        labels: number of labels possible
      Returns:
        one-hot encoded dense Tensor [batch_size, k].
      """

      with tf.name_scope('target'):
        tensor = tf.convert_to_tensor(targets)

        hot_rows = tf.cast(
            tf.expand_dims(
                tf.range(0, batch_size), 1), tensor.dtype)
        hot_indices = tf.concat(concat_dim=1, values=[hot_rows, tensor])
        shape = tf.cast(tf.pack([batch_size, labels]), tensor.dtype)

        return tf.sparse_to_dense(hot_indices,
                                  shape,
                                  sparse_values=tf.constant(1,
                                                            dtype=tf.float32),
                                  default_value=tf.constant(0,
                                                            dtype=tf.float32))

    with tf.name_scope('inputs'):
      examples = tf.placeholder(tf.string,
                                shape=self.hyperparams.batch_size,
                                name='examples')
      parsed_examples = tf.parse_example(examples, self._features)
      self.examples = examples
      self.targets = parsed_examples['label']
      # Replace with one_hot when it becomes available.
      self.encoded_targets = encode_targets(
          self.targets, self.hyperparams.batch_size, self.hyperparams.labels)
      self.embeddings = parsed_examples['embedding']
      self.keys = parsed_examples['image_uri']

  def _create_layer(self, layer, weights, biases):
    new_layer = tf.nn.xw_plus_b(layer, weights, biases)
    return new_layer, tf.nn.relu(new_layer)

  def _create_hidden_layers(self):
    # Create two sets of layers with shared weights,
    # one for prediction and one for training. This allows us
    # use droput in training but not for prediction.
    train_layer = self.embeddings
    prediction_layer = self.embeddings
    last_layer_size = self.hyperparams.embedding_size
    layers = [self.hyperparams.hidden_layer_size, self.hyperparams.labels]
    for i, layer_size in enumerate(layers):
      with tf.name_scope('layer%s' % i):
        train_layer = tf.nn.dropout(train_layer,
                                    self.hyperparams.dropout_keep_prob)
        weights = tf.Variable(
            tf.truncated_normal(
                [last_layer_size, layer_size],
                stddev=1.0 / math.sqrt(float(last_layer_size))),
            name='weights')
        biases = tf.Variable(tf.zeros([layer_size]), name='biases')
        train_new_layer, train_layer = self._create_layer(train_layer,
                                                          weights, biases)
        prediction_new_layer, prediction_layer = self._create_layer(
            prediction_layer, weights, biases)
        last_layer_size = layer_size
    return train_new_layer, prediction_new_layer

  def _create_outputs(self, logits):
    with tf.name_scope('outputs'):
      self.scores = tf.nn.softmax(logits, name='scores')
      # Choose the top one as prediction
      self.predictions = tf.arg_max(logits, 1, name='prediction')

  def _create_training(self, logits):
    with tf.name_scope('train'):
      entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                        self.encoded_targets)
      self.loss = tf.reduce_mean(entropy, name='loss')

      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      optimizer = tf.train.AdagradOptimizer(0.01)
      self.train = optimizer.minimize(self.loss, self.global_step)

  def _create_initialization(self):
    self.initialize = tf.initialize_all_variables()
