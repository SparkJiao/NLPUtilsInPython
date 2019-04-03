from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
import json
import argparse
from collections import Counter
import random
from copy import deepcopy
from tqdm import tqdm
import numpy as np

tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True, ner=True))

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--sample_rate', type=float, default=1.0)
args = parser.parse_args()

pronouns = ['he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'that', 'those', 'this',
            'these', 'there', 'here']
cut_length = 12
min_length = 5
ner_tag_cnt = 3
pos_tag_cnt = 3


def question_label(question):
    # True for <START>, False for others
    q_tokens = tokenizer.tokenize(question)

    # Contains pronouns
    for q_token in q_tokens:
        if q_token.text in pronouns:
            return False

    # Question length exceed max question length
    if len(q_tokens) > cut_length:
        return True

    if len(q_tokens) < min_length:
        return False

    # Entity type and POS tag number exceed threshold in a question
    ner_counter = Counter()
    pos_counter = Counter()
    for q_token in q_tokens:
        ner_counter[q_token.ent_type_] += 1
        pos_counter[q_token.pos_] += 1
    if len(ner_counter) > ner_tag_cnt:
        return True
    if len(pos_counter) > pos_tag_cnt:
        return True

    return False


""" This only shuffle train set which does't has the 'additional_answers' key. """

with open(args.input_file, 'r') as f:
    data_file = json.load(f)

if 'data' in data_file:
    data = data_file['data']
else:
    data = data_file

print('Length of initial data set: {}'.format(len(data)))

for article in tqdm(data, desc='Labeling questions...'):
    questions = article['questions']
    for question in questions:
        if question_label(question['input_text']):
            question['label'] = 'START'
        else:
            question['label'] = 'INSIDE'

output_data = []

for article in tqdm(data, desc='Shuffling questions in group...'):

    art_cp = deepcopy(article)
    do_switch = False

    questions = article['questions']
    answers = article['answers']
    if 'additional_answers' in article:
        add_answers = article['additional_answers']
    else:
        add_answers = None
    pre = -1
    for i, question in enumerate(questions):
        if i < pre:
            continue
        if question['label'] == 'START':
            for j in range(i + 1, len(questions)):
                if questions[j]['label'] == 'START' or j == len(questions) - 1:
                    pre = j

                    idx_ls = range(i + 1, j)
                    if len(idx_ls) == 0:
                        break
                    # sample_idx = random.sample(idx_ls, len(idx_ls))
                    sample_idx = np.random.permutation(idx_ls)
                    turn_id_s = questions[i + 1]['turn_id']
                    turn_id_e = questions[j - 1]['turn_id']

                    sample_questions = [questions[idx] for idx in sample_idx]
                    for idx, sample in enumerate(sample_questions):
                        questions[i + 1 + idx] = sample
                        questions[i + 1 + idx]['turn_id'] = turn_id_s + idx
                    sample_answers = [answers[idx] for idx in sample_idx]
                    for idx, sample in enumerate(sample_answers):
                        answers[i + 1 + idx] = sample
                        answers[i + 1 + idx]['turn_id'] = turn_id_s + idx
                    assert questions[i + len(idx_ls)]['turn_id'] == turn_id_e

                    do_switch = True

    output_data.append(art_cp)
    if do_switch:
        sample_rand = random.randint(0, 100) / 100.0
        if sample_rand <= args.sample_rate:
            article['id'] += '$'
            output_data.append(article)

print('Length of new data set: {}'.format(len(output_data)))

output_json = {'data': output_data}
with open(args.output_file, 'w') as f:
    json.dump(output_json, f, indent=2)
