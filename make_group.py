import argparse
import json
import nltk.tokenize as tk
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', dest='input_file')
parser.add_argument('--output-file', dest='output_file')
parser.add_argument('--cut_length', type=int, default=11)
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    dataset = json.load(f)
data = dataset['data']
version = dataset['version']

output_ls = []
group_num = Counter()

num_a = 0
num_b = 0

for article in data:
    q = article['questions']
    a = article['answers']
    num_a += len(article['questions'])
    questions = []
    answers = []
    groups = 0
    for idx, (que, ans) in enumerate(zip(q, a)):
        if len(tk.word_tokenize(que['input_text'])) > args.cut_length:
            if len(questions) > 0:
                output_ls.append({'source': article['source'],
                                  'id': article['id'],
                                  'filename': article['filename'],
                                  'story': article['story'],
                                  'questions': questions,
                                  'answers': answers,
                                  'name': article['name']})
                num_b += len(questions)

            groups += 1

            questions = [que]
            answers = [ans]
        else:
            questions.append(que)
            answers.append(ans)
    output_ls.append({'source': article['source'],
                      'id': article['id'],
                      'filename': article['filename'],
                      'story': article['story'],
                      'questions': questions,
                      'answers': answers,
                      'name': article['name']})
    num_b += len(questions)
    groups += 1
    group_num[groups] += 1

with open(args.output_file, 'w') as f:
    json.dump({'version': version, 'data': output_ls}, f, indent=2)

print(json.dumps(group_num, indent=2))
print(num_a, num_b)
