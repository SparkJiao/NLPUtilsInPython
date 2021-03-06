from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
import json
import argparse
from collections import Counter
import random
from copy import deepcopy
from tqdm import tqdm

tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True, ner=True))

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
parser.add_argument('--switch_rate', type=float, default=0.4)
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


with open(args.input_file, 'r') as f:
    data_file = json.load(f)

if 'data' in data_file:
    data = data_file['data']
else:
    data = data_file

print('Length of initial data set: {}'.format(len(data)))

for article in data:
    questions = article['questions']
    for question in questions:
        if question_label(question['input_text']):
            question['label'] = 'START'
        else:
            question['label'] = 'INSIDE'

output_data = []

for article in tqdm(data):

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
        if i <= pre:
            continue
        if question['label'] == 'START':
            for j in range(i + 1, len(questions)):
                if questions[j]['label'] == 'START' or j == len(questions) - 1:
                    pre = j

                    idx_ls = range(i + 1, j)
                    if len(idx_ls) == 0:
                        break
                    sample_num = int(len(idx_ls) * args.switch_rate)
                    sample_idx = random.sample(idx_ls, sample_num)

                    for idx in sample_idx:
                        if idx + 1 < len(questions):
                            a_turn_id = questions[idx]['turn_id']
                            b_turn_id = questions[idx + 1]['turn_id']

                            questions[idx], questions[idx + 1] = questions[idx + 1], questions[idx]
                            questions[idx]['turn_id'] = a_turn_id
                            questions[idx + 1]['turn_id'] = b_turn_id
                            answers[idx], answers[idx + 1] = answers[idx + 1], answers[idx]
                            answers[idx]['turn_id'] = a_turn_id
                            answers[idx + 1]['turn_id'] = b_turn_id
                            if add_answers is not None:
                                for x in add_answers:
                                    add_answers[x][idx], add_answers[x][idx + 1] = add_answers[x][idx + 1], add_answers[x][idx]
                                    add_answers[x][idx]['turn_id'] = a_turn_id
                                    add_answers[x][idx + 1]['turn_id'] = b_turn_id

                            do_switch = True

    output_data.append(art_cp)
    if do_switch:
        article['id'] += '$'
        output_data.append(article)


print('Length of new data set: {}'.format(len(output_data)))

output_json = {'data': output_data}
with open(args.output_file, 'w') as f:
    json.dump(output_json, f, indent=2)
