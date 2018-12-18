import json
import argparse
from CoQA_eval import CoQAEvaluator
from collections import Counter


def judge(text):
    if text == 'yes':
        return 1
    if text == 'no':
        return 2
    if text == 'unknown':
        return 0
    return 3


def get_yesno_recall_precision(predicted_answers, gold_answers):
    eva = []
    precision = []
    recall = []
    for i in range(4):
        a = []
        for j in range(4):
            a.append(0)
        eva.append(a)
    # 0:unknown 1:yes 2:no 3:x
    for i, (pd, gd) in enumerate(zip(predicted_answers, gold_answers)):
        a = Counter([judge(text.strip().lower()) for text in gd]).most_common(1)[0][0]
        b = judge(pd.strip().lower())
        eva[a][b] += 1
    for i in range(4):
        tp = float(eva[i][i])
        fp = 0.0
        for j in range(4):
            if j != i:
                fp += eva[j][i]
        fn = 0.0
        for j in range(4):
            if j != i:
                fn += eva[i][j]
        if tp + fp != 0:
            precision.append(tp / (tp + fp))
        else:
            precision.append(-1)
        if tp + fn != 0:
            recall.append(tp / (tp + fn))
        else:
            recall.append(-1)
    yes = {'precision': precision[1], 'recall': recall[1]}
    no = {'precision': precision[2], 'recall': recall[2]}
    x = {'precision': precision[3], 'recall': recall[3]}
    unknown = {'precision': precision[0], 'recall': recall[0]}
    out = {'yes': yes, 'no': no, 'x': x, 'unknown': unknown}
    return out


parser = argparse.ArgumentParser('description: experiments on datasets')
parser.add_argument('data_file')
parser.add_argument('pred_file')
parser.add_argument('--turn_id', default=[1], nargs='*', type=int)
args = parser.parse_args()

turn_ids = args.turn_id

with open(args.data_file, 'r') as f:
    data = json.load(f)['data']

with open(args.pred_file, 'r') as f:
    predictions = json.load(f)

preds = {}
for prediction in predictions:
    story_id = prediction['id']
    if story_id not in preds:
        preds[story_id] = {}
    turn_id = prediction['turn_id']
    answer = prediction['answer']
    preds[story_id][turn_id] = answer

evals = dict()
predicted_answers = []
gold_answers = []
for turn_id in turn_ids:
    turn = 'turn_id: ' + str(turn_id)
    golds = []
    pds = []
    for article in data:
        if turn_id > len(article['questions']):
            continue
        gold = [article['answers'][turn_id - 1]['input_text']]
        gold += [article['additional_answers'][key][turn_id - 1]['input_text'] for key in article['additional_answers']]
        golds.append(gold)

        pds.append(preds[article['id']][turn_id])

        print('gold_answer: %s\nprediction: %s\n' % (gold, pds[-1]))

    evals[turn] = dict()
    evals[turn]['F1'] = CoQAEvaluator.compute_turn_score_seq(golds, pds)

    output = get_yesno_recall_precision(pds, golds)

    evals[turn].update(output)
    # print(json.dumps(evals[turn_id], indent=4))

    predicted_answers.extend(pds)
    gold_answers.extend(golds)

print('=============================Single Turn Evaluation==========================================')

print(json.dumps(evals, indent=4))

print('==================================All Dev Set================================================')

evals['all dev set'] = dict()
evals['all dev set']['F1'] = CoQAEvaluator.compute_turn_score_seq(gold_answers, predicted_answers)
evals['all dev set'].update(get_yesno_recall_precision(predicted_answers, gold_answers))

print(json.dumps(evals['all dev set'], indent=4))
