import argparse
import json
import random
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_dir')
args = parser.parse_args()

opt = vars(args)

summary_writer = SummaryWriter(log_dir=opt['output_dir'])

with open(opt['input_file'], 'r') as f:
    for line in f:
        line = line.split()
        if 'dev' in line:
            # print(line)
            # print(int(line[3])-1, line[7])
            summary_writer.add_scalar('eval/F1', float(line[7]) * 0.01, int(line[3]) - 1)

summary_writer.close()
