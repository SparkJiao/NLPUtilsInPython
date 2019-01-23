from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
import json
import argparse
from collections import Counter

tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True, ner=True))

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
args = parser.parse_args()

pronouns = ['he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'that', 'those', 'this',
            'these', 'there', 'here']
cut_length = 12
min_length = 5
ner_tag_cnt = 3
pos_tag_cnt = 3


with open(args.input_file, 'r') as f:
    data_file = json.load(f)

if 'data' in data_file:
    data = data_file['data']

    for article in data:
        questions = article['questions']

else:
    data = data_file


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

