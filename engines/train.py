# -*- coding: utf-8 -*-
# @Time : 2020/10/22 9:14 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import tensorflow as tf
import math
import time
from tqdm import tqdm
from tensorflow_addons.text.crf import crf_decode
from transformers import TFBertModel, BertTokenizer
from engines.models import BiLSTM_CRFModel, DomainClassificationModel, IntentClassificationModel
from engines.utils.metrics import cal_metrics, cal_slots_metrics


def train(configs, data_manager, logger):
    domain_classes = data_manager.domain_class_number
    intent_classes = data_manager.intent_class_number
    slot_classes = data_manager.slot_class_number
    id2slot = data_manager.id2slot
    learning_rate = configs.learning_rate
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
            X_train_batch, att_mask_train_batch, domain_train_batch, intent_train_batch, slot_train_batch \
                = data_manager.next_batch(X_train, att_mask_train, domain_train, intent_train, slot_train,
                                          start_index=iteration * batch_size)
            inputs_length = tf.math.count_nonzero(X_train_batch, 1)
            # 获得bert模型的输出
            bert_model_inputs = bert_model(X_train_batch, attention_mask=att_mask_train_batch)[0]
            with tf.GradientTape() as tape:
                # 槽位模型输入
                slot_logits, slot_log_likelihood, slot_transition_params = bilstm_crf_model.call(
                    inputs=bert_model_inputs, inputs_length=inputs_length, targets=slot_train_batch, training=1)
                slot_loss = -tf.reduce_mean(slot_log_likelihood)
                # 主题模型的输入
                domain_logits = domain_model.call(inputs=bert_model_inputs[:, 0, :], training=1)
                domain_loss_vec = tf.keras.losses.sparse_categorical_crossentropy(y_pred=domain_logits,
                                                                                  y_true=domain_train_batch)
                domain_loss = tf.reduce_mean(domain_loss_vec)
                # 意图模型的输入
                intent_logits = intent_model.call(inputs=bert_model_inputs[:, 0, :], training=1)
                intent_loss_vec = tf.keras.losses.sparse_categorical_crossentropy(y_pred=intent_logits,
                                                                                  y_true=intent_train_batch)
                intent_loss = tf.reduce_mean(intent_loss_vec)
                total_loss = domain_loss + intent_loss + 2 * slot_loss
            # 参数列表
            trainable_variables = bilstm_crf_model.trainable_variables + domain_model.trainable_variables + intent_model.trainable_variables
            # 定义好参加梯度的参数
            gradients = tape.gradient(total_loss, trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            if iteration % configs.print_per_batch == 0 and iteration != 0:
                domain_predictions = tf.argmax(domain_logits, axis=-1)
                intent_predictions = tf.argmax(intent_logits, axis=-1)
                domain_measures = cal_metrics(y_true=domain_train_batch, y_pred=domain_predictions)
                intent_measures = cal_metrics(y_true=intent_train_batch, y_pred=intent_predictions)
                batch_pred_sequence, _ = crf_decode(slot_logits, slot_transition_params, inputs_length)
                slot_measures = cal_slots_metrics(X_train_batch, slot_train_batch, batch_pred_sequence, id2slot, tokenizer)
                domain_str = ''
                for k, v in domain_measures.items():
                    domain_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, domain_loss: %.5f, %s' % (iteration, domain_loss, domain_str))
                intent_str = ''
                for k, v in intent_measures.items():
                    intent_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, intent_loss: %.5f, %s' % (iteration, intent_loss, intent_str))
                slot_str = ''
                for k, v in slot_measures.items():
                    slot_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, slot_loss: %.5f, %s' % (iteration, slot_loss, slot_str))
        # validation
        logger.info('start evaluate engines...')
        slot_val_results = {'precision': 0, 'recall': 0, 'f1': 0}
        domain_val_results = {'precision': 0, 'recall': 0, 'f1': 0}
        intent_val_results = {'precision': 0, 'recall': 0, 'f1': 0}
        for iteration in tqdm(range(num_val_iterations)):
            X_val_batch, att_mask_val_batch, domain_val_batch, intent_val_batch, slot_val_batch \
                = data_manager.next_batch(X_val, att_mask_val, domain_val, intent_val, slot_val,
                                          start_index=iteration * batch_size)
            inputs_length = tf.math.count_nonzero(X_val_batch, 1)
            # 获得bert模型的输出
            bert_model_inputs = bert_model(X_val_batch, attention_mask=att_mask_val_batch)[0]
            # 槽位模型预测
            slot_logits, slot_log_likelihood, slot_transition_params = bilstm_crf_model.call(
                inputs=bert_model_inputs, inputs_length=inputs_length, targets=slot_val_batch)
            batch_pred_sequence, _ = crf_decode(slot_logits, slot_transition_params, inputs_length)
            slot_measures = cal_slots_metrics(X_val_batch, slot_val_batch, batch_pred_sequence, id2slot, tokenizer)
            # 主题模型的预测
            domain_logits = domain_model.call(inputs=bert_model_inputs[:, 0, :])
            domain_predictions = tf.argmax(domain_logits, axis=-1)
            domain_measures = cal_metrics(y_true=domain_val_batch, y_pred=domain_predictions)
            # 意图模型的预测
            intent_logits = intent_model.call(inputs=bert_model_inputs[:, 0, :])
            intent_predictions = tf.argmax(intent_logits, axis=-1)
            intent_measures = cal_metrics(y_true=intent_val_batch, y_pred=intent_predictions)

            for k, v in slot_measures.items():
                slot_val_results[k] += v
            for k, v in domain_measures.items():
                domain_val_results[k] += v
            for k, v in intent_measures.items():
                intent_val_results[k] += v

        time_span = (time.time() - start_time) / 60
        val_slot_str = ''
        val_domain_str = ''
        val_intent_str = ''
        for k, v in slot_val_results.items():
            slot_val_results[k] /= num_val_iterations
            val_slot_str += (k + ': %.3f ' % slot_val_results[k])
        for k, v in domain_val_results.items():
            domain_val_results[k] /= num_val_iterations
            val_domain_str += (k + ': %.3f ' % domain_val_results[k])
        for k, v in intent_val_results.items():
            intent_val_results[k] /= num_val_iterations
            val_intent_str += (k + ': %.3f ' % intent_val_results[k])
        logger.info(val_slot_str)
        logger.info(val_domain_str)
        logger.info(val_intent_str)
        logger.info('time consumption:%.2f(min)' % time_span)
