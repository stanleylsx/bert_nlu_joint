# -*- coding: utf-8 -*-
# @Time : 2020/9/28 4:14 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py 
# @Software: PyCharm
import os
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm


class DataManager:
    """
    Bert的数据管理器
    """
    def __init__(self, configs, logger):
        self.configs = configs
        self.train_file = configs.train_file
        self.logger = logger
        self.hyphen = configs.hyphen

        self.train_file = configs.datasets_fold + '/' + configs.train_file

        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.label_scheme = configs.label_scheme
        self.label_level = configs.label_level
        self.suffix = configs.suffix
        self.PADDING = '[PAD]'

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.vocabs_dir = configs.vocabs_dir
        self.label2id_file = self.vocabs_dir + '/label2id'
        self.label2id, self.id2label = self.load_labels()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.max_token_number = len(self.tokenizer.get_vocab())
        self.max_label_number = len(self.label2id)

    def load_labels(self):
        """
        若不存在词表则生成，若已经存在则加载词表
        :return:
        """
        if not os.path.isfile(self.label2id_file):
            self.logger.info('label vocab files not exist, building label vocab...')
            return self.build_labels(self.train_file)

        self.logger.info('loading label vocab...')
        label2id, id2label = {}, {}
        with open(self.label2id_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label
        return label2id, id2label

    def build_labels(self, train_path):
        """
        根据训练集生成词表
        :param train_path:
        :return:
        """
        df_train = read_csv(train_path, names=['token', 'label'], delimiter=self.configs.delimiter)
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        label2id = dict(zip(labels, range(1, len(labels) + 1)))
        id2label = dict(zip(range(1, len(labels) + 1), labels))
        # 向生成的词表和标签表中加入[PAD]
        id2label[0] = self.PADDING
        label2id[self.PADDING] = 0
        # 保存标签表
        with open(self.label2id_file, 'w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')
        return label2id, id2label

    def next_batch(self, X, y, att_mask, start_index):
        """
        下一次个训练批次
        :param X:
        :param y:
        :param att_mask:
        :param start_index:
        :return:
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        att_mask_batch = list(att_mask[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
                att_mask_batch.append(att_mask[index])
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        att_mask_batch = np.array(att_mask_batch)
        return X_batch, y_batch, att_mask_batch

    def prepare(self, df):
        self.logger.info('loading data...')
        X = []
        y = []
        att_mask = []
        tmp_x = []
        tmp_y = []
        for index, record in tqdm(df.iterrows()):
            token = record.token
            label = record.label
            if str(token) == str(np.nan):
                if len(tmp_x) <= self.max_sequence_length - 2:
                    tmp_x = self.tokenizer.encode(tmp_x)
                    tmp_att_mask = [1] * len(tmp_x)
                    tmp_y = [self.label2id[y] for y in tmp_y]
                    tmp_y.insert(0, self.label2id['O'])
                    tmp_y.append(self.label2id['O'])
                    # padding
                    tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                    tmp_y += [self.label2id[self.PADDING] for _ in range(self.max_sequence_length - len(tmp_y))]
                    tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                    X.append(tmp_x)
                    y.append(tmp_y)
                    att_mask.append(tmp_att_mask)
                else:
                    # 此处的padding不能在self.max_sequence_length加2，否则不同维度情况下，numpy没办法转换成矩阵
                    tmp_x = tmp_x[:self.max_sequence_length-2]
                    tmp_x = self.tokenizer.encode(tmp_x)
                    X.append(tmp_x)
                    tmp_y = tmp_y[:self.max_sequence_length-2]
                    tmp_y = [self.label2id[y] for y in tmp_y]
                    tmp_y.insert(0, self.label2id['O'])
                    tmp_y.append(self.label2id['O'])
                    y.append(tmp_y)
                    tmp_att_mask = [1] * self.max_sequence_length
                    att_mask.append(tmp_att_mask)
                tmp_x = []
                tmp_y = []
            else:
                tmp_x.append(token)
                tmp_y.append(label)
        return np.array(X), np.array(y), np.array(att_mask)

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        :param train_val_ratio:
        :return:
        """
        df_train = read_csv(self.train_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        X, y, att_mask = self.prepare(df_train)
        # shuffle the samples
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        att_mask = att_mask[indices]

        if self.dev_file is not None:
            X_train = X
            y_train = y
            att_mask_train = att_mask
            X_val, y_val, att_mask_val = self.get_valid_set()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            att_mask_train = att_mask[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
            att_mask_val = att_mask[int(num_samples * train_val_ratio):]
            self.logger.info('validating set is not exist, built...')
        self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(y_val)))
        return X_train, y_train, att_mask_train, X_val, y_val, att_mask_val

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = read_csv(self.dev_file, names=['token', 'label'], delimiter=self.configs.delimiter)
        X_val, y_val, att_mask_val = self.prepare(df_val)
        return X_val, y_val, att_mask_val

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        sentence = list(sentence)
        if len(sentence) <= self.max_sequence_length - 2:
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
            x += [0 for _ in range(self.max_sequence_length - len(x))]
            att_mask += [0 for _ in range(self.max_sequence_length - len(att_mask))]
        else:
            sentence = sentence[:self.max_sequence_length-2]
            x = self.tokenizer.encode(sentence)
            att_mask = [1] * len(x)
        y = [self.label2id['O']] * self.max_sequence_length
        return np.array([x]), np.array([y]), np.array([att_mask]), np.array([sentence])
