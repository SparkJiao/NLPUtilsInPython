import argparse
import json
from collections import Counter
from util import AverageMeter
from CoQA_eval import CoQAEvaluator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--predict_file')
parser.add_argument('--dev_file')
args = parser.parse_args()

opt = vars(args)

with open(opt['predict_file'], 'r') as f:
    predictions = json.load(f)

with open(opt['dev_file'], 'r') as f:
    dev = json.load(f)

evaluator = CoQAEvaluator(dev)

span_start_dis_cnt = Counter()
span_end_dis_cnt = Counter()
wrong_type = 0

dis_diff_cnt = dict()
f1_length = dict()

matching_more_set = []

for data in predictions:
    spans = data['spans']
    truth_s = spans[0]
    truth_e = spans[1]
    predict_s = spans[2]
    predict_e = spans[3]

    if (truth_s == -1 and predict_s != -1) or (truth_s != -1 and predict_s == -1):
        wrong_type += 1
        continue

    g_length = truth_e - truth_s + 1
    if g_length not in dis_diff_cnt:
        dis_diff_cnt[g_length] = (Counter(), Counter())
    if g_length not in f1_length:
        f1_length[g_length] = AverageMeter()

    # if g_length == 0:
    #     print(data)

    if predict_s < truth_s and predict_e > truth_e:
        matching_more_set.append({'prediction': data['answer'], 'answer': evaluator.gold_data[(data['id'], data['turn_id'])]})

    span_start_dis_cnt[truth_s - predict_s] += 1
    span_end_dis_cnt[truth_e - predict_e] += 1

    dis_diff_cnt[g_length][0][truth_s - predict_s] += 1
    dis_diff_cnt[g_length][1][truth_e - predict_e] += 1

    f1_length[g_length].update(evaluator.compute_turn_score(data['id'], data['turn_id'], data['answer'])['f1'], 1)

print('wrong type predict: {} / {} = {}'.format(wrong_type, len(predictions), wrong_type * 1.0 / len(predictions)))

print('F1 distribution on gold answer length(most common = 5):')

f1_cnt = Counter()
for ans_len in f1_length:
    f1_cnt[ans_len] = f1_length[ans_len].avg
print(json.dumps(f1_cnt.most_common(5), indent=2))

print('F1 distribution on gold answer length(all):')
f1_dict = {ans_len: f1_length[ans_len].avg for ans_len in f1_length}
sorted_f1_length = sorted(f1_dict.items(), key=lambda d: d[0])
for ans_len in sorted_f1_length:
    print('gold answer length: {}, F1: {}, questions num: {}'.format(ans_len[0], ans_len[1], f1_length[ans_len[0]].count))

print('span start distance different:')
print(json.dumps(span_start_dis_cnt.most_common(5), indent=2))
print('span end distance different:')
print(json.dumps(span_end_dis_cnt.most_common(5), indent=2))

print('Difference for each answer length:')
for i in range(1, 10):
    print(f'Answer length = {i}')
    print('span start distance difference:')
    print(json.dumps(dis_diff_cnt[i][0].most_common(5), indent=2))
    print('span end distance difference:')
    print(json.dumps(dis_diff_cnt[i][1].most_common(5), indent=2))

print('Print the prediction which has a larger scale than gold answer randomly 30:')

idx_perm = range(0, len(matching_more_set))
idx_perm = np.random.permutation(idx_perm)
output = []
for i in range(30):
    output.append(matching_more_set[idx_perm[i]])
print(json.dumps(output, indent=2))

