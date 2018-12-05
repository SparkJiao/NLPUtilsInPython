import json
import argparse


def allennlp_dialog_qa_transfer(input_file, output_file):
    with open(input_file) as f:
        data = list()
        for line in f:
            data.append(json.loads(line))
    outputs = list()
    for pred in data:
        story_id = pred['id']
        best_span_str = pred['best_span_str']
        qid_list = pred['qid']
        yesno_list = pred['yesno']
        for (best_span, turn_id, yesno) in zip(best_span_str, qid_list, yesno_list):
            if yesno == 'y':
                answer = 'yes'
            elif yesno == 'n':
                answer = 'no'
            else:
                answer = best_span
            outputs.append({
                'id': story_id,
                'turn_id': turn_id,
                'answer': answer,
                'yesno': yesno,
                'best_span': best_span
            })
    with open(output_file, 'w') as file:
        json.dump(outputs, file, indent=2)
    print('transfer complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('description: transfer model prediction file to official prediction file')
    parser.add_argument('index', type=int, help='index for transfer function')
    parser.add_argument('--input-file', dest='input_file')
    parser.add_argument('--output-file', dest='output_file')
    parser.add_argument('--num', type=int, default=-1)
    args = parser.parse_args()

    if args.index == 0:
        allennlp_dialog_qa_transfer(args.input_file, args.output_file)
    else:
        raise RuntimeError('bad experiment name, see help')
