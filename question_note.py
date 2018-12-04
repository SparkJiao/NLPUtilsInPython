import json
from sys import argv


def squad_1():
    pass


def coqa(input_file, output_path, train_or_dev):
    with open(input_file, 'r') as f:
        data = json.load(f)['data']
    output = list()
    for article in data:
        questions = article['questions']
        for i, question in enumerate(questions):
            if i == 0:
                output.append({'question': question['input_text'], 'flag': 'y'})
            else:
                output.append({'question': question['input_text'], 'flag': 'n'})
    with open(output_path + '/coqa-preprocess-' + train_or_dev + '.json', 'w') as f:
        json.dump(output, f, indent=4)


def quac(input_file, output_path, train_or_dev):
    with open(input_file, 'r') as f:
        pass

def squad_2(input_file, output_path, train_or_dev):
    with open(input_file, 'r') as f:
        data = json.load(f)['data']
    output = list()
    for article in data:
        qas = article['paragraph']['qas']
        for qa in qas:
            question = qa['question']
            output.append({'question': question, 'flag': 'y'})
    with open(output_path + '/squad-v2.0-preprocess-' + train_or_dev + '.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    dataset = argv[1]
    input_file = argv[2]
    output_path = argv[3]
    train_or_dev = argv[4]
    if train_or_dev != 'train' and train_or_dev != 'dev':
        raise RuntimeError("Bad train_or_dev flag, need \'train\'/\'dev\'")
    if dataset == 'squad1.1':
        squad_1()
    elif dataset == 'squad2.0':
        squad_2(input_file, output_path, train_or_dev)
    elif dataset == 'coqa':
        coqa(input_file, output_path, train_or_dev)
    elif dataset == 'quac':
        quac(input_file, output_path, train_or_dev)
    else:
        raise RuntimeError("Bad dataset, need \'squad1.1\'/\'squad2.0\'/\'coqa\'/\'quac\'")
