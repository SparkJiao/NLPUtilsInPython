import argparse
from collections import Counter
import json
from CoQA_eval import CoQAEvaluator
import logging
import re
import string

logger = logging.getLogger()

parser = argparse.ArgumentParser('description: experiments on datasets')
parser.add_argument('input_file')
parser.add_argument('output_file')
args = parser.parse_args()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))


def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', s)])


def free_text_to_span(free_text, full_text):
    if free_text == "unknown":
        return "__NA__", -1, -1
    if normalize_answer(free_text) == "yes":
        return "__YES__", -1, -1
    if normalize_answer(free_text) == "no":
        return "__NO__", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls) - 1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i + j >= len(full_ls): break
            full_cnt[full_ls[i + j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert (best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i + best_j][1] + 1

    return full_text[char_i:char_j], char_i, char_j


def proc_train(ith, article):
    rows = []

    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            answers = qa['orig_answer']

            answer = answers['text']
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])
            answer_choice = 0 if answer == 'CANNOTANSWER' else \
                            1 if qa['yesno'] == 'y' else \
                            2 if qa['yesno'] == 'n' else \
                            3  # Not a yes/no question
            if answer_choice == 0:
                answer_start, answer_end = -1, -1
            rows.append((ith, question, answer, answer_start, answer_end, answer_choice, paragraph['id'], qa['id']))
    return rows, context


out = []
with open(args.input_file, 'r') as f:
    data = json.load(f)['data']
    for i, article in enumerate(data):
        rows, context = proc_train(i, article)

        q_text = [row[1] for row in rows]
        answer = [row[2] for row in rows]
        answer_start = [row[3] for row in rows]
        answer_end = [row[4] for row in rows]
        answer_choice = [row[5] for row in rows]
        paragraph_id = [row[6] for row in rows]
        question_id = [row[7] for row in rows]
        out.append(
            {'context': context, 'q_text': q_text, 'answer': answer, 'answer_start': answer_start,
             'answer_end': answer_end, 'answer_choice': answer_choice, 'paragraph_id': paragraph_id, 'question_id': question_id})

with open(args.output_file, 'w') as f:
    json.dump(out, f, indent=2)
