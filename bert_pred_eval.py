import argparse
import json
from CoQA_eval import CoQAEvaluator
import logging

logger = logging.getLogger()

parser = argparse.ArgumentParser('description: experiments on datasets')
parser.add_argument('--pred_file')
parser.add_argument('--data_file')
args = parser.parse_args()

with open(args.pred_file, 'r') as f:
    data = json.load(f)
    predictions = []
    for answer in data:
        predictions.append(data[answer])

with open(args.data_file, 'r') as f:
    data = json.load(f)['data']
    ground_truth = []
    for article in data:
        answers = [[answer['input_text']] for answer in article['answers']]
        add_answers = article['additional_answers']
        for key in add_answers:
            for i, additional_answer in enumerate(add_answers[key]):
                answers[i].append(additional_answer['input_text'])
        ground_truth.extend(answers)

F1 = CoQAEvaluator.compute_turn_score_seq(ground_truth, predictions)

print("F1: %f" % F1)
