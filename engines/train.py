# -*- coding: utf-8 -*-
# @Time : 2020/10/22 9:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import math
import time
from tqdm import tqdm
from tensorflow_addons.text.crf import crf_decode
from transformers import TFBertModel, BertTokenizer
from engines.models import BiLSTM_CRFModel, DomainClassificationModel, IntentClassificationModel


def train(configs, data_manager, logger):
    domain_classes = data_manager.domain_class_number
    intent_classes = data_manager.intent_class_number
    slot_classes = data_manager.slot_class_number
    learning_rate = configs.learning_rate
    max_to_keep = configs.checkpoints_max_to_keep
    checkpoints_dir = configs.checkpoints_dir
    checkpoint_name = configs.checkpoint_name
    best_f1_val = 0.0
    best_at_epoch = 0
    unprocessed = 0
    very_start_time = time.time()
    epoch = configs.epoch
    batch_size = configs.batch_size

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    bert_model = TFBertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    X_train, att_mask_train, domain_train, intent_train, slot_train, \
    X_val, att_mask_val, domain_val, intent_val, slot_val = data_manager.get_training_set()

    bilstm_crf_model = BiLSTM_CRFModel(configs, slot_classes)
    domain_model = DomainClassificationModel(configs, domain_classes)
    intent_model = IntentClassificationModel(configs, intent_classes)

    num_iterations = int(math.ceil(1.0 * len(X_train) / batch_size))
    num_val_iterations = int(math.ceil(1.0 * len(X_val) / batch_size))
    logger.info(('+' * 20) + 'training starting' + ('+' * 20))

    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for iteration in tqdm(range(num_iterations)):
            X_train_batch, att_mask_train_batch, domain_train_batch, intent_train_batch, slot_train_batch\
                = data_manager.next_batch(X_train, att_mask_train, domain_train, intent_train, slot_train,
                                          start_index=iteration * batch_size)
            inputs_length = tf.math.count_nonzero(X_train_batch, 1)
            # 获得bert模型的输出
            bert_model_inputs = bert_model(X_train_batch, attention_mask=att_mask_train_batch)[0]
            with tf.GradientTape() as tape:
                # 槽位模型输入
                slot_logits, slot_log_likelihood, slot_transition_params = bilstm_crf_model.call(
                    inputs=bert_model_inputs, inputs_length=inputs_length, targets=slot_train, training=1)
                slot_loss = -tf.reduce_mean(slot_log_likelihood)
                # 主题模型的输入
                domain_logits = domain_model.call(inputs=bert_model_inputs)
                domain_loss_vec = tf.keras.losses.sparse_categorical_crossentropy(y_pred=domain_logits,
                                                                                  y_true=domain_train)
                domain_loss = tf.reduce_mean(domain_loss_vec)
                # 意图模型的输入
                intent_logits = intent_model.call(inputs=bert_model_inputs)
                intent_loss_vec = tf.keras.losses.sparse_categorical_crossentropy(y_pred=intent_logits,
                                                                                  y_true=intent_train)
                intent_loss = tf.reduce_mean(intent_loss_vec)
                total_loss = domain_loss + intent_loss + 2 * slot_loss
                # 定义好参加梯度的参数
                gradients = tape.gradient(total_loss, )


