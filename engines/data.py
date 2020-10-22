# -*- coding: utf-8 -*-
# @Time : 2020/9/28 4:14 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py 
# @Software: PyCharm
import os
import re
import json
import pickle
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm


class DataManager:
    """
    数据管理器
    """

    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.train_file = configs.datasets_fold + '/' + configs.train_file
        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.PADDING = '[PAD]'
        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.vocabs_dir = configs.vocabs_dir
        self.labels_file = self.vocabs_dir + '/labels'

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.id2domain, self.domain2id, self.id2intent, self.intent2id, self.id2slot, self.slot2id = self.load_labels()
        self.max_token_number = len(self.tokenizer.get_vocab())
        self.domain_class_number = len(self.domain2id)
        self.intent_class_number = len(self.intent2id)
        self.slot_class_number = len(self.slot2id)

    @staticmethod
    def read_data(files):
        """read json data """
        with open(files, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_labels(self):
        """
        若不存在标签映射则生成，若已经存在则加载标签映射
        :return:
        """
        if not os.path.isfile(self.labels_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        with open(self.labels_file, 'rb') as f:
            return pickle.load(f)

    def build_labels(self, file_path):
        """
        根据训练集生成标签映射
        :param file_path:
        :return:
        """
        data_list = self.read_data(file_path)
        domain_labels = set([data['domain'] for data in data_list])
        intent_labels = set([str(data['intent']) for data in data_list])
        slots_labels = set()
        for data in tqdm(data_list):
            for slot in data['slots']:
                slots_labels.add('B-{}'.format(slot))
                slots_labels.add('I-{}'.format(slot))
        slots_labels = list(slots_labels)
        id2domain = {i: label for i, label in enumerate(domain_labels)}
        domain2id = {label: i for i, label in id2domain.items()}

        id2intent = {i: label for i, label in enumerate(intent_labels)}
        intent2id = {label: i for i, label in id2intent.items()}

        id2slot = {i: label for i, label in enumerate(slots_labels, 4)}
        id2slot[0] = self.PADDING
        id2slot[1] = '[CLS]'
        id2slot[2] = '[SEP]'
        id2slot[3] = 'O'
        slot2id = {label: i for i, label in id2slot.items()}
        # 保存
        with open(self.labels_file, 'wb') as fw:
            pickle.dump([id2domain, domain2id, id2intent, intent2id, id2slot, slot2id], fw)
        return id2domain, domain2id, id2intent, intent2id, id2slot, slot2id

    def next_batch(self, X, att_mask, domain, intent, slot, start_index):
        """
        下一次个训练批次
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        att_mask_batch = list(att_mask[start_index:min(last_index, len(X))])
        domain_batch = list(domain[start_index:min(last_index, len(X))])
        intent_batch = list(intent[start_index:min(last_index, len(X))])
        slot_batch = list(slot[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                att_mask_batch.append(att_mask[index])
                domain_batch.append(domain[index])
                intent_batch.append(intent[index])
                slot_batch.append(slot[index])
        return np.array(X_batch), np.array(att_mask_batch), np.array(domain_batch), np.array(intent_batch), np.array(slot_batch)

    def get_slot_label(self, text, slots):
        tag = ['O'] * len(text)
        for k, v in slots.items():
            search = re.search(v, text)
            if search:
                start, end = search.start(), search.end()
                tag[start] = 'B-{}'.format(k)
                for index in range(start + 1, end):
                    tag[index] = 'I-{}'.format(k)
        tag_id = [self.slot2id[t] for t in tag]
        return tag_id

    def prepare(self, data_list):
        self.logger.info('loading data...')
        X = []
        domain = []
        intent = []
        slot = []
        att_mask = []
        for data in tqdm(data_list):
            text = data['text']
            tmp_domain = self.domain2id[data['domain']]
            tmp_intent = self.intent2id[data['intent']]
            domain.append(tmp_domain)
            intent.append(tmp_intent)
            if len(text) <= self.max_sequence_length - 2:
                tmp_x = self.tokenizer.encode(text)
                tmp_att_mask = [1] * len(tmp_x)
                tmp_slot = self.get_slot_label(text, data['slots'])
                tmp_slot.insert(0, self.slot2id['O'])
                tmp_slot.append(self.slot2id['O'])
                # padding
                tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                tmp_slot += [self.slot2id[self.PADDING] for _ in range(self.max_sequence_length - len(tmp_slot))]
                X.append(tmp_x)
                att_mask.append(tmp_att_mask)
                slot.append(tmp_slot)
            else:
                text = text[:self.max_sequence_length - 2]
                tmp_x = self.tokenizer.encode(text)
                tmp_att_mask = [1] * self.max_sequence_length
                tmp_slot = self.get_slot_label(text, data['slots'])
                tmp_slot.insert(0, self.slot2id['O'])
                tmp_slot.append(self.slot2id['O'])
                X.append(tmp_x)
                att_mask.append(tmp_att_mask)
                slot.append(tmp_slot)
        return np.array(X), np.array(att_mask), np.array(domain), np.array(intent), np.array(slot)

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        data_list = self.read_data(self.train_file)
        X, att_mask, domain, intent, slot = self.prepare(data_list)
        # shuffle the samples
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        att_mask = att_mask[indices]
        domain = domain[indices]
        intent = intent[indices]
        slot = slot[indices]

        if self.dev_file is not None:
            X_train = X
            att_mask_train = att_mask
            domain_train = domain
            intent_train = intent
            slot_train = slot
            X_val, att_mask_val, domain_val, intent_val, slot_val = self.get_valid_set()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            att_mask_train = att_mask[:int(num_samples * train_val_ratio)]
            domain_train = domain[:int(num_samples * train_val_ratio)]
            intent_train = intent[:int(num_samples * train_val_ratio)]
            slot_train = slot[:int(num_samples * train_val_ratio)]

            X_val = X[int(num_samples * train_val_ratio):]
            att_mask_val = att_mask[int(num_samples * train_val_ratio):]
            domain_val = domain[int(num_samples * train_val_ratio):]
            intent_val = intent[int(num_samples * train_val_ratio):]
            slot_val = slot[int(num_samples * train_val_ratio):]
            self.logger.info('validating set is not exist, built...')
        self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(X_val)))
        return X_train, att_mask_train, domain_train, intent_train, slot_train, X_val, att_mask_val, domain_val, intent_val, slot_val

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        data_list = self.read_data(self.dev_file)
        X_val, att_mask_val, domain_val, intent_val, slot_val = self.prepare(data_list)
        return X_val, att_mask_val, domain_val, intent_val, slot_val

    # def prepare_single_sentence(self, sentence):
    #     """
    #     把预测的句子转成矩阵和向量
    #     :param sentence:
    #     :return:
    #     """
    #     sentence = list(sentence)
    #     if len(sentence) <= self.max_sequence_length - 2:
    #         x = self.tokenizer.encode(sentence)
    #         att_mask = [1] * len(x)
    #         x += [0 for _ in range(self.max_sequence_length - len(x))]
    #         att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
    #     else:
    #         sentence = sentence[:self.max_sequence_length-2]
    #         x = self.tokenizer.encode(sentence)
    #         att_mask = [1] * len(x)
    #     y = [self.label2id['O']] * self.max_sequence_length
    #     return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])
