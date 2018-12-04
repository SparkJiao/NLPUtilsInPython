import json
import argparse

def pronoun_mask(input_file, output_file):
    """
    挑出问题中包含代词的问题并把代词去掉
    """
    with open(input_file, 'r') as f:
        data = json.load(f)['data']
    for article in data:
        questions = article['questions']
        for question in questions:
            q_text = question['input_text']
            n_q_text = q_text.replace('he')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description: experiments on datasets')
    parser.add_argument('experiment', help='pronoun: pronoun_mask(input-file, output-file)\n')
    parser.add_argument('--input-file', dest='input_file')
    parser.add_argument('--output-file', dest='output_file')
    args = parser.parse_args()

    if args.experiment == 'pronoun':
        pronoun_mask(args.input_file, args.output_file)
    else:
        raise RuntimeError('bad experiment name, see help')