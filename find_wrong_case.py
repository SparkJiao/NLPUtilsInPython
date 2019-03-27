import json
import argparse
from CoQA_eval import CoQAEvaluator
from util import AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--with_flow')
parser.add_argument('--no_flow')
parser.add_argument('--dev')
parser.add_argument('--output_dir')
args = parser.parse_args()

opt = vars(args)

with open(opt['with_flow'], 'r') as f:
    with_flow = json.load(f)

with open(opt['no_flow'], 'r') as f:
    no_flow = json.load(f)

with open(opt['dev'], 'r') as f:
    dev_set = json.load(f)

evaluator = CoQAEvaluator(dev_set)

dev_map = dict()

for data in dev_set['data']:
    questions = data['questions']
    answers = data['answers']
    for question, answer in zip(questions, answers):
        dev_map[(data['id'], question['turn_id'])] = (question['input_text'], answer['input_text'])

flow_higher = []
no_flow_higher = []
equality = []

max_f1 = AverageMeter()

for h, nh in zip(with_flow, no_flow):
    assert h['id'] == nh['id'] and h['turn_id'] == nh['turn_id']

    with_flow_f1 = evaluator.compute_turn_score(h['id'], h['turn_id'], h['answer'])['f1']
    no_flow_f1 = evaluator.compute_turn_score(nh['id'], nh['turn_id'], nh['answer'])['f1']

    story_id = h['id']
    turn_id = h['turn_id']
    flow_answer = h['answer']
    no_flow_answer = nh['answer']
    question, gold = dev_map[(story_id, turn_id)]

    max_f1.update(max(with_flow_f1, no_flow_f1), 1)

    ent = {
        'id': story_id,
        'turn_id': turn_id,
        'question': question,
        'flow_answer': flow_answer,
        'no_flow_answer': no_flow_answer,
        'gold': gold
    }

    if with_flow_f1 > no_flow_f1:
        flow_higher.append(ent)
    elif with_flow_f1 < no_flow_f1:
        no_flow_higher.append(ent)
    else:
        equality.append(ent)

print(f'Length table:\nflow higher: {len(flow_higher)}, no_flow_higher: {len(no_flow_higher)}, equality: {len(equality)}')
print('If only calculate the max F1 score of both model, the final F1 score will be {}', max_f1.avg)

with open(opt['output_dir'] + '/flow_higher.json', 'w') as f:
    json.dump(flow_higher, f, indent=2)
with open(opt['output_dir'] + '/no_flow_higher.json', 'w') as f:
    json.dump(no_flow_higher, f, indent=2)
with open(opt['output_dir'] + '/euqality.json', 'w') as f:
    json.dump(equality, f, indent=2)
