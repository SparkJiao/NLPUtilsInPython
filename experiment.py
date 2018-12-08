import json
import argparse
import re


def replace_pronoun(s, r):
    pattern = re.compile(r'[^a-zA-Z]he[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]him[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]she[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]her[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]it[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]its[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]they[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    pattern = re.compile(r'[^a-zA-Z]them[^a-zA-Z]', re.I)
    s = pattern.sub(r, s)
    return s


def pronoun_mask(input_file, output_file, num):
    """
    挑出问题中包含代词的问题并把代词去掉
    """
    with open(input_file, 'r') as f:
        data = json.load(f)['data']
    for article in data:
        questions = article['questions']
        for question in questions:
            q_text = question['input_text']
            n_q_text = replace_pronoun(q_text, ' ')
            question['input_text'] = n_q_text
        if num != -1:
            article['questions'] = questions[:num]
            article['answers'] = article['answers'][:num]
            if 'additional_answers' in article:
                for key in article['additional_answers']:
                    article['additional_answers'][key] = article['additional_answers'][key][:num]
    with open(output_file, 'w') as f:
        json.dump({'data': data}, f, indent=2)


def pick_sentence_lack_elements(input_file, output_file):
    """
    单独择出某些文章，其中的部分问题句子成分残缺
    """
    id_list = [
        '37xitheisw95z8hh4d6i4n863aecrr',
        '3c2nj6jbkah7msxned0vjquaphjn21',
        '3bv8hq2zzw1okamzsb7tnxrm7kfa6p',
        '3qxnc7eipivf1gqfygdci16bnvn90b',
        '3wrfbplxraow7at6ide020z2w3tn33',
        '34j10vatjfyw0aohj8d4a0wwjhgiqt',
        '33tin5lc04acybm06oolat0v0f99yy'
    ]
    print("id_list:")
    for idx in id_list:
        print(idx)
    with open(input_file, 'r') as f:
        data = json.load(f)['data']
    output = []
    for article in data:
        if article['id'] not in id_list:
            continue
        else:
            output.append(article)
    with open(output_file, 'w') as f:
        json.dump({'data': output}, f, indent=2)
    print("finish")


def pick_questions(input_file, output_file, num):
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    data = dataset['data']
    for article in data:
        questions = article['questions'][:num]
        article['questions'] = questions
        answers = article['answers'][:num]
        article['answers'] = answers
        additional_answers = article['additional_answers']
        for key in additional_answers:
            article['additional_answers'][key] = additional_answers[key][:num]
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print('finish')


def pick_different(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        data1 = json.load(f)
    with open(input_file2, 'r') as f:
        data2 = json.load(f)
    output = list()
    for (p1, p2) in zip(data1, data2):
        assert p1['id'] == p2['id']
        assert p1['turn_id'] == p2['turn_id']
        if p1['predicted_yesno'] != p2['predicted_yesno'] or p1['best_span'] != p2['best_span']:
            uncom = dict()
            com = dict()
            uncom['question'] = p1['question']
            com['question'] = p2['question']
            uncom['best_span'] = p1['best_span']
            com['best_span'] = p2['best_span']
            uncom['yesno'] = p1['predicted_yesno']
            com['yesno'] = p2['predicted_yesno']
            output.append({
                'id': p1['id'],
                'turn_id': p1['turn_id'],
                'answer': p1['answer_text'],
                'additional_answers': p1['additional_answers'],
                'uncom': uncom,
                'com': com
            })
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


def combine_question(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        questions = json.load(f)['data']
    with open(input_file2, 'r') as f:
        preds = json.load(f)
    tmp = list()
    for article in questions:
        story_id = article['id']
        questions = article['questions']
        answers = article['answers']
        additional_answers = article['additional_answers']
        for i, (question, answer) in enumerate(zip(questions, answers)):
            output = dict()
            output['id'] = story_id
            output['question'] = question['input_text']
            output['turn_id'] = question['turn_id']
            output['answer_text'] = answer['input_text']
            output['additional_answers'] = list()
            for key in additional_answers:
                output['additional_answers'].append(additional_answers[key][i]['input_text'])
            tmp.append(output)
    outputs = list()
    for (inp, pred) in zip(tmp, preds):
        assert inp['id'] == pred['id']
        assert inp['turn_id'] == pred['turn_id']
        output = dict()
        output['id'] = pred['id']
        output['turn_id'] = pred['turn_id']
        output['question'] = inp['question']
        output['predicted_answer'] = pred['answer']
        output['predicted_yesno'] = pred['yesno']
        output['best_span'] = pred['best_span']
        output['answer_text'] = inp['answer_text']
        output['additional_answers'] = inp['additional_answers']
        outputs.append(output)
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=2)
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description: experiments on datasets')
    parser.add_argument('experiment', help='pronoun: pronoun_mask(input-file, output-file)\n')
    parser.add_argument('--input-file', dest='input_file')
    parser.add_argument('--output-file', dest='output_file')
    parser.add_argument('--input-file2', dest='input_file2')
    parser.add_argument('--num', type=int, default=-1)
    args = parser.parse_args()

    if args.experiment == 'pronoun':
        pronoun_mask(args.input_file, args.output_file, args.num)
    elif args.experiment == 'pick':
        pick_sentence_lack_elements(args.input_file, args.output_file)
    elif args.experiment == 'pick-questions':
        pick_questions(args.input_file, args.output_file, args.num)
    elif args.experiment == 'pick-different':
        pick_different(args.input_file, args.input_file2, args.output_file)
    elif args.experiment == 'combine':
        combine_question(args.input_file, args.input_file2, args.output_file)
    else:
        raise RuntimeError('bad experiment name, see help')
