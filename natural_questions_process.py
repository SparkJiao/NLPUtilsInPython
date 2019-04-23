import json
import jsonlines
import argparse
import os
import gzip
from tqdm import tqdm
from collections import Counter
import random
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_clean_tokens(nq_document_tokens):
    return [token['token'].replace(" ", "") for token in nq_document_tokens if not token['html_token']]


def judge_yes_no(answer_list):
    cnt = Counter()
    for answer in answer_list:
        cnt[answer] += 1
    answer = cnt.most_common(1)[0][0]
    assert answer in ['YES', 'NO', 'NONE']
    return answer


def all_path(dirname):
    result = []  # 所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            result.append(apath)

    return result


def depress_files(sub_args):
    opt = vars(sub_args)

    data_dir = opt['data_dir']

    data_files = all_path(data_dir)

    for data_file in data_files:
        cmd = 'gzip -d {}'.format(data_file)
        os.system(cmd)

    print('Done.')


def extract_yesno_examples(sub_args):
    opt = vars(sub_args)

    data_dir = opt['data_dir']
    output_file = opt['output_file']
    do_split = opt['do_split']

    data_files = all_path(data_dir)

    output_list = []

    for data_file in data_files:
        # with open(data_file, 'r++', encoding='utf-8') as f:
        with gzip.open(data_file, 'rb') as f:
            json_lines = jsonlines.Reader(f)

            # For details of data format, please see:
            # https://github.com/google-research-datasets/natural-questions
            # Dict:
            # annotations
            #   dict_keys(['annotation_id', 'long_answer', 'short_answers', 'yes_no_answer'])
            # document_html
            # document_title
            # document_tokens
            # document_url
            # example_id
            # long_answer_candidates
            # question_text
            # question_tokens

            for index, item in tqdm(enumerate(json_lines), desc='Processing data...'):
                is_yesno = False
                an_id = -1
                annotations = item['annotations']
                for idx, annotate in enumerate(annotations):
                    if annotate['yes_no_answer'] in ['YES', 'NO']:
                        is_yesno = True
                        an_id = idx
                        break

                if is_yesno:
                    if not do_split:
                        output_list.append(item)
                    else:
                        span_s, span_e = annotations[an_id]['long_answer']['start_byte'], annotations[an_id]['long_answer']['end_byte']
                        token_s, token_e = \
                            annotations[an_id]['long_answer']['start_token'], annotations[an_id]['long_answer']['end_token']
                        if span_s != -1 and span_e != -1:
                            item['document_html'] = item['document_html'][span_s: span_e]
                            item['document_tokens'] = item['document_tokens'][token_s: token_e]
                        else:
                            item['document_html'] = ''
                            item['document_tokens'] = []
                        output_list.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        # json.dump(output_list, f, indent=2)
        json.dump(output_list, f)
    print('Done.')


def extract_yesno_examples_longer_text(sub_args):
    opt = vars(sub_args)

    data_dir = opt['data_dir']
    train_ratio = opt['train_ratio']
    max_doc_len = opt['max_doc_len']
    output_dir = opt['output_dir']

    logger.info(f'Train set ratio: {train_ratio}')
    logger.info(f'Max paragraph length: {max_doc_len}')

    data_files = all_path(data_dir)

    train_list = []
    dev_list = []

    for data_file in data_files:
        # with open(data_file, 'r++', encoding='utf-8') as f:
        with gzip.open(data_file, 'rb') as f:
            json_lines = jsonlines.Reader(f)

            # For details of data format, please see:
            # https://github.com/google-research-datasets/natural-questions
            # Dict:
            # annotations
            #   dict_keys(['annotation_id', 'long_answer', 'short_answers', 'yes_no_answer'])
            # document_html, removed
            # document_title, removed
            # document_tokens
            # document_url, removed
            # example_id
            # long_answer_candidates, removed
            # question_text
            # question_tokens

            for index, item in tqdm(enumerate(json_lines), desc='Processing data...'):
                annotations = item['annotations']

                yes_no_list = [x['yes_no_answer'] for x in annotations]
                yes_no = judge_yes_no(yes_no_list)

                if yes_no == 'NONE':
                    continue

                ans_id = -1
                for idx, yes_no_text in enumerate(yes_no_list):
                    if yes_no == yes_no_text:
                        ans_id = idx

                init_doc_tokens = item['document_tokens']

                token_s, token_e = annotations[ans_id]['long_answer']['start_token'], annotations[ans_id]['long_answer']['end_token']
                doc_tokens = get_clean_tokens(init_doc_tokens[token_s: token_e])

                if len(doc_tokens) < max_doc_len:
                    pending = (max_doc_len - len(doc_tokens)) // 2
                    bef_pend_list = []
                    aft_pend_list = []

                    bef_p = token_s - 1
                    while len(bef_pend_list) < pending and bef_p >= 0:
                        if not init_doc_tokens[bef_p]['html_token']:
                            bef_pend_list.append(init_doc_tokens[bef_p]['token'].replace(" ", ""))
                        bef_p -= 1

                    aft_p = token_e
                    while len(aft_pend_list) < pending and aft_p < len(init_doc_tokens):
                        if not init_doc_tokens[aft_p]['html_token']:
                            aft_pend_list.append(init_doc_tokens[aft_p]['token'].replace(" ", ""))
                        aft_p += 1

                    doc_tokens = bef_pend_list + doc_tokens + aft_pend_list

                    # Currently we don't add evidence token start and end label to example.
                    # token_s = len(bef_pend_list)
                    # token_e = token_s + len(doc_tokens)

                example = {'example_id': item['example_id'],
                           'question_text': item['question_text'],
                           'document_tokens': doc_tokens,
                           'yes_no_answer': yes_no}
                # 'start_token': token_s,
                # 'end_token': token_e}
                if random.random() < train_ratio:
                    train_list.append(example)
                else:
                    dev_list.append(example)

    logger.info('Generate train Yes/No examples: {}'.format(len(train_list)))
    logger.info('Generate dev Yes/No examples: {}'.format(len(dev_list)))

    with open(os.path.join(output_dir, 'train-yesno-examples.json'), 'w', encoding='utf-8') as f:
        json.dump(train_list, f, indent=2)
    with open(os.path.join(output_dir, 'dev-yesno-examples.json'), 'w', encoding='utf-8') as f:
        json.dump(dev_list, f, indent=2)
    logger.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process data of Natural Questions')

    sub_parser = parser.add_subparsers(title='sub-commands', description='name of different process.')

    depress = sub_parser.add_parser('depress', help='Depress all .gz files to initial .jsonl files.')
    depress.add_argument('--data_dir', type=str, help='Directory of all data files.')
    depress.set_defaults(func=depress_files)

    yesno_extractor = sub_parser.add_parser('yesno', help='Extract examples which have yes/no answers.')
    yesno_extractor.add_argument('--data_dir', type=str, help='Directory of all data files.')
    yesno_extractor.add_argument('--output_file', type=str, help='Output json file for yesno examples.')
    yesno_extractor.add_argument('--do_split', action='store_true', default=False,
                                 help='If extract the tokens and html for the long answer annotation.')
    yesno_extractor.set_defaults(func=extract_yesno_examples)

    get_yesno_examples = sub_parser.add_parser('yesno_example', help='Generate examples with yes/no answers and max document length.')
    get_yesno_examples.add_argument('--data_dir', type=str, help='Directory of all data files.')
    get_yesno_examples.add_argument('--output_dir', type=str, help='Output directory for yesno examples.')
    get_yesno_examples.add_argument('--train_ratio', type=float, default=0.8)
    get_yesno_examples.add_argument('--max_doc_len', type=int, default=150)
    get_yesno_examples.set_defaults(func=extract_yesno_examples_longer_text)

    args = parser.parse_args()
    args.func(args)
