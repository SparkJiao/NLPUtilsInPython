import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ext-pred', type=str, help='the extractive prediction file')
parser.add_argument('abs-pred', type=str, help='the abstraction prediction file')
parser.add_argument('-out-file', type=str, help='the output file')

args = parser.parse_args()

with open(args.ext_pred, 'w') as f:
    ext_pred = json.load(f)
with open(args.abs_pred, 'w') as f:
    abs_pred = json.load(f)

outputs = list()
assert len(ext_pred) == len(abs_pred)

for ex, ab in zip(ext_pred, abs_pred):
    assert ex['id'] == ab['id']
    assert ex['turn_id'] == ab['turn_id']
    output = dict()
    output['id'] = ex['id']
    output['turn_id'] = ex['turn_id']
    yesno = ex['yesno']
    if yesno == 'a':
        output['answer'] = 'yes'
    elif yesno == 'x':
        output['answer'] = ab['answer']