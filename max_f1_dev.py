import spacy
from collections import Counter
import json
from spacy.tokens import Token
import logging

logger = logging.getLogger()

nlp = spacy.load('en')


def get_max_f1_span(sentences, input_text):
    answer = Counter(input_text)

    best_sentence = None
    sen_f1 = 0.0
    best_sen_index = -1
    for index, sentence in enumerate(sentences):
        sen_cnt = Counter(sentence)
        common = answer & sen_cnt
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(sentence)
        recall = 1.0 * num_same / len(input_text)
        f1 = (2 * precision * recall) / (precision + recall)
        if sen_f1 < f1:
            sen_f1 = f1
            best_sentence = sentence
            best_sen_index = index
        # print(index, f1)

    if best_sen_index == -1:
        print(input_text)
        return -1, -1

    base = 0
    for i in range(best_sen_index):
        base += len(sentences[i])
    start = base
    end = base + len(best_sentence) - 1

    return start, end


with open('./coqa-dev-v1.0.json', 'r', encoding="utf8") as f:
    data = json.load(f)['data']
    # story = data[0]['story'].lower()
    # tokens = nlp(story)
    # for token in tokens:
    #     print(token)

num_qa = 0
f1_total = 0
for i, article in enumerate(data):
    # id = article['id']
    story = article['story'].lower().replace('\n', ' ')
    tokens = nlp(story)
    # print(tokens)
    sents = nlp(story).sents
    sentences = []
    for sent in sents:
        tmp = []
        for token in sent:
            tmp.append(token.text)
        sentences.append(tmp)

    answers = article['answers']
    for answer in answers:
        r = answer['input_text'].lower()
        if r == 'yes' or r == 'no' or r == 'unknown':
            span = (-1, -1)
            f1 = 1.0
            pred = r
        else:
            input_text_token = nlp(r)
            input_text = [token.text for token in input_text_token]
            # print(input_text)
            span = get_max_f1_span(sentences, input_text)
            # print(span)
            if span == (-1, -1):
                pred = r
            else:
                pred = tokens[span[0]:(span[1] + 1)].text
        # f1_total += f1
        # print(pred)
        answer['pred'] = str(pred)
        answer['pred_start'] = span[0]
        answer['pred_end'] = span[1]
    num_qa += len(answers)
    print("%d / %d" % (i, len(data)))

# final_f1 = f1_total / num_qa
# print("F1: %s" % (str(final_f1)))

with open('coqa-pred.json', 'w') as f:
    json.dump(data, f, indent=4)
