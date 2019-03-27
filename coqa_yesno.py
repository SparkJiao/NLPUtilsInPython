import json
import argparse
from CoQA_eval import CoQAEvaluator
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict_file')
parser.add_argument('-d', '--dev_file')

args = parser.parse_args()
opt = vars(args)

with open(opt['dev_file'], 'r') as f:
    dev_file = json.load(f)
evaluator = CoQAEvaluator(dev_file)


def judge_yesno(gold_list):
    yesno_dict = Counter()
    for gold in gold_list:
        norm_text = CoQAEvaluator.normalize_answer(gold)
        if norm_text == 'yes':
            yesno_dict['y'] += 1
        elif norm_text == 'no':
            yesno_dict['n'] += 1
        elif norm_text == 'unknown':
            yesno_dict['u'] += 1
        else:
            yesno_dict['x'] += 1
    return yesno_dict.most_common(1)[0][0]


def judge_f1(gold_list, pred_list, positive_list, negative_list):
    cnt = Counter()
    for (gold, pred) in zip(gold_list, pred_list):
        if pred in positive_list and gold in positive_list:
            cnt['true_positive'] += 1
        elif pred in positive_list and gold in negative_list:
            cnt['false_positive'] += 1
        elif pred in negative_list and gold in positive_list:
            cnt['false_negative'] += 1
        elif pred in negative_list and gold in negative_list:
            cnt['true_negative'] += 1
    return cnt


def calculate_f1(true_positive, false_positive, true_negative, false_negative):
    TP = true_positive
    FP = false_positive
    TN = true_negative
    FN = false_negative
    dic = {
        'precision': TP * 1.0 / (TP + FP),
        'recall': TP * 1.0 / (TP + FN),
        'accuracy': (TP + TN) * 1.0 / (TP + FP + TN + FN),
    }
    dic['F1'] = 2 * dic['precision'] * dic['recall'] / (dic['precision'] + dic['recall'])
    return dic


with open(opt['predict_file'], 'r') as f:
    data = json.load(f)

yesno_cnt = Counter()
yesno_set = ['y', 'n', 'u', 'x']
gold_type_list = []
pred_type_list = []

for pred in data:
    s_id = pred['id']
    turn_id = pred['turn_id']
    prediction = pred['answer']

    golds = evaluator.gold_data[(s_id, turn_id)]

    gold_type = judge_yesno(golds)
    pred_type = judge_yesno([prediction])

    yesno_cnt[(gold_type, pred_type)] += 1
    gold_type_list.append(gold_type)
    pred_type_list.append(pred_type)

# All results
for key in yesno_cnt:
    print('{}: {}'.format(key, yesno_cnt[key]))

# Accuracy on (yes, no, unknown)/not
is_yesno = ['y', 'n', 'u']  # positive
not_yesno = ['x']  # negative
is_cnt = judge_f1(gold_type_list, pred_type_list, is_yesno, not_yesno)

print('Accuracy on yes/no/unknown(positive) and not(negative):')
print(json.dumps(is_cnt, indent=2))
print(json.dumps(calculate_f1(is_cnt['true_positive'], is_cnt['false_positive'], is_cnt['true_negative'], is_cnt['false_negative']), indent=2))

