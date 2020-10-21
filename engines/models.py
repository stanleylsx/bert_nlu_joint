# -*- coding: utf-8 -*-
# @Time : 2020/9/28 2:58 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : models.py
# @Software: PyCharm
from abc import ABC

import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood


class BiLSTM_CRFModel(tf.keras.Model, ABC):
    def __init__(self, configs, vocab_size, num_classes):
        super(BiLSTM_CRFModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, configs.embedding_dim, mask_zero=True)
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_classes)
        self.transition_params = tf.Variable(tf.random.uniform(shape=(num_classes, num_classes)))

    @tf.function
    def call(self, inputs, inputs_length, targets, training=None):
        embedding_inputs = inputs
        dropout_inputs = self.dropout(embedding_inputs, training)
        bilstm_outputs = self.bilstm(dropout_inputs)
        logits = self.dense(bilstm_outputs)
        tensor_targets = tf.convert_to_tensor(targets, dtype=tf.int64)
        log_likelihood, self.transition_params = crf_log_likelihood(
            logits, tensor_targets, inputs_length, transition_params=self.transition_params)
        return logits, log_likelihood, self.transition_params


class DomainClassificationModel(tf.keras.Model, ABC):
    def __init__(self, num_classes):
        super(DomainClassificationModel, self).__init__()
        self.dense = tf.keras.layers.Dense(num_classes)

    @tf.function
    def call(self, inputs):
        embedding_inputs = inputs
        logits = self.dense(embedding_inputs)
        return logits


class IntentClassificationModel(tf.keras.Model, ABC):
    def __init__(self, num_classes):
        super(IntentClassificationModel, self).__init__()
        self.dense = tf.keras.layers.Dense(num_classes)

    @tf.function
    def call(self, inputs):
        embedding_inputs = inputs
        logits = self.dense(embedding_inputs)
        return logits




