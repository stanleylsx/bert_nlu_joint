# -*- coding: utf-8 -*-
# @Time : 2020/10/23 9:51 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm

from sklearn import metrics


def get_slot_name(text, slot_label):
    slots = {}
    for i, slot in enumerate(slot_label):
        if slot == 'O':
            continue
        else:
            _, slot_name = slot.split('-')
            if slot_name in slots:
                slots[slot_name] += text[i]
            else:
                slots[slot_name] = text[i]
    return slots


def cal_metrics(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    return {'precision': precision, 'recall': recall, 'f1': f1}


def cal_slots_metrics(y_true, y_pred):
    pass

