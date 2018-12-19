import argparse
import json
from CoQA_eval import CoQAEvaluator

parser = argparse.ArgumentParser('description: experiments on datasets')
parser.add_argument('data_file')
parser.add_argument('pred_file')
args = parser.parse_args()

with open(args.data_file, 'r') as f:
    data_file = json.load(f)['data']
    data = []
    for article in data_file:
        story_id = article['id']
        answers = article['answers']
        additional_answers = article['additional_answers']
        for i, answer in enumerate(answers):
            tmp = [answer['input_text']]
            tmp.extend([additional_answers[key][i]['input_text'] for key in additional_answers])
            data.append(tmp)

with open(args.pred_file, 'r') as f:
    pred_file = json.load(f)

output = []
for i, (gold, pre) in enumerate(zip(data, pred_file)):
    f1 = CoQAEvaluator._compute_turn_score(gold, pre['answer'])['f1']
    if f1 <= 0.7:
        output.append({'id': pre['id'], 'turn_id': pre['turn_id'], 'gold': gold, 'pred': pre['answer']})

print(json.dumps(output, indent=2))
