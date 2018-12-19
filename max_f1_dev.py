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


def proc_dev(ith, article):
    rows = []
    context = article['story']

    for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
        gold_answer = answers['input_text']
        span_answer = answers['span_text']

        answer, char_i, char_j = free_text_to_span(gold_answer, span_answer)
        answer_choice = 0 if answer == '__NA__' else \
            1 if answer == '__YES__' else \
                2 if answer == '__NO__' else \
                    3  # Not a yes/no question

        if answer_choice == 3:
            answer_start = answers['span_start'] + char_i
            answer_end = answers['span_start'] + char_j
        else:
            answer_start, answer_end = -1, -1

        rationale = answers['span_text']
        rationale_start = answers['span_start']
        rationale_end = answers['span_end']

        q_text = question['input_text']
        # if j > 0:
        #     q_text = article['answers'][j - 1]['input_text'] + " // " + q_text

        rows.append(
            (ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
    return rows, context


out = []
with open(args.input_file, 'r', encoding="utf8") as f:
    data = json.load(f)['data']
    golds = []
    ex = []
    for i, article in enumerate(data):
        rows, context = proc_dev(i, article)

        answers = article['answers']
        if 'additional_answers' in article:
            add_answers = article['additional_answers']
        else:
            add_answers = {}
        for j, answer in enumerate(article['answers']):
            tmp = list()
            tmp.append(answer['input_text'])
            for key in add_answers:
                tmp.append(add_answers[key][j]['input_text'])
            golds.append(tmp)
        e_tmp = list()
        for row in rows:
            if row[2] == '__NA__':
                e_tmp.append('unknown')
            elif row[2] == '__YES__':
                e_tmp.append('yes')
            elif row[2] == '__NO__':
                e_tmp.append('no')
            else:
                e_tmp.append(row[2])
        ex.extend(e_tmp)

        q_text = [row[1] for row in rows]
        answer = [row[2] for row in rows]
        answer_start = [row[3] for row in rows]
        answer_end = [row[4] for row in rows]
        rationale = [row[5] for row in rows]
        rationale_start = [row[6] for row in rows]
        rationale_end = [row[7] for row in rows]
        answer_choice = [row[8] for row in rows]
        out.append(
            {'context': context, 'story_id': article['id'], 'q_text': q_text, 'answer': answer, 'answer_start': answer_start,
             'answer_end': answer_end, 'rationale': rationale, 'rationale_start': rationale_start, 'rationale_end': rationale_end,
             'answer_choice': answer_choice})

    F1 = CoQAEvaluator.compute_turn_score_seq(golds, ex)
    print('F1: %f' % F1)

with open(args.output_file, 'w') as f:
    json.dump(out, f, indent=2)
