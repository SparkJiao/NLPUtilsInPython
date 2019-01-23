import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_dir')
parser.add_argument('--split_num', type=int, default=10)
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    json_data = json.load(f)
    if 'data' in json_data:
        data = json_data['data']
    else:
        data = json_data

output_set_t = []

for i in range(args.split_num):
    output_set_t.append([])

for article in data:
    t_index = random.randint(0, args.split_num - 1)
    output_set_t[t_index].append(article)

for i in range(args.split_num):
    with open(args.output_dir + '/train-set-split-' + str(i) + '.json', 'w') as f:
        json.dump(output_set_t[i], f, indent=2)
    print("length of train set %d is: %d" % (i, len(output_set_t[i])))
