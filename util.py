import torch
from torch import Tensor


def combine_bert_piece(bert_embedding: Tensor):
    """
    :param bert_embedding: piece_count * max_seq_length *
    :return:
    """
    pass


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
