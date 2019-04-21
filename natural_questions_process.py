import json
import jsonlines
import argparse
import os
import gzip
from tqdm import tqdm


def all_path(dirname):
    result = []  #所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  #合并成一个完整路径
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
                        token_s, token_e = annotations[an_id]['long_answer']['start_token'], annotations[an_id]['long_answer']['end_token']
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

    args = parser.parse_args()
    args.func(args)
