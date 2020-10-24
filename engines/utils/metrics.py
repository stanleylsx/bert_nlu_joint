# -*- coding: utf-8 -*-
# @Time : 2020/10/23 9:51 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py
# @Software: PyCharm
from sklearn import metrics


def cal_metrics(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    return {'precision': precision, 'recall': recall, 'f1': f1}


def get_slots_item(text, slot_label, id2slot):
    slots = {}
    text = [token for token in text if token != '[PAD]']
    slot_label = slot_label[:len(text)]
    slots_token = [id2slot[slot] for slot in slot_label]
    for i, slot in enumerate(slots_token):
        if slot == 'O':
            continue
        else:
            _, slot_name = slot.split('-')
            if slot_name in slots:
                slots[slot_name] += text[i]
            else:
                slots[slot_name] = text[i]
    return slots


def cal_slots_metrics(x_batch, y_true_batch, y_pred_batch, id2slot, tokenizer):
    correct, p_denominator, r_denominator = 0, 0, 0
    precision = -1.0
    recall = -1.0
    f1 = -1.0
    for x, y_true, y_pred in zip(x_batch, y_true_batch, y_pred_batch):
        text = tokenizer.convert_ids_to_tokens(x.tolist())
        true_dic = get_slots_item(text, y_true.tolist(), id2slot)
        pred_dic = get_slots_item(text, y_pred.numpy().tolist(), id2slot)
        r_denominator += len(true_dic)
        p_denominator += len(pred_dic)
        for key, value in pred_dic.items():
            if key not in pred_dic:
                continue
            elif pred_dic[key] == pred_dic[key]:
                correct += 1
    if p_denominator != 0:
        precision = float(correct) / p_denominator
    if r_denominator != 0:
        recall = float(correct) / r_denominator
    if precision > 0 and recall > 0:
        f1 = 2 * precision * recall / (precision + recall) * 1.0
    return {'precision': precision, 'recall': recall, 'f1': f1}

