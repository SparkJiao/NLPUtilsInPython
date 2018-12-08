import torch
from torch import nn
from allennlp.nn import util


class LinearSelfAtt(nn.Module):
    def __init__(self, input_dim):
        super(LinearSelfAtt, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        compute self-attention vector
            gamma = softmax(w^T X)
            r = sum(gamma_i * X_i)
        :param x: b * m * h
        :param mask: b * m
        :return: r: b * h
        """
        # b * m * 1
        gamma = util.masked_softmax(self.linear(x).squeeze(2), mask)
        # [b * 1 * m] * [b * m * h] = [b * 1 * h]
        # b * h
        r = gamma.unsqueeze(1).bmm(x).squeeze(1)
        return r


class MultiHeadSelfAtt(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=4):
        super(MultiHeadSelfAtt, self).__init__()
        self.multi_head = nn.ModuleList()
        self.num_head = head_num
        for i in range(head_num):
            self.multi_head.append(LinearSelfAtt(input_dim))
        self.linear = nn.Linear(input_dim * head_num, output_dim)

    def forward(self, x, mask=None):
        """
        compute multi-head self-attention
        :param x: b * m * h
        :param mask: b * m
        :return: b * output_dim
        """
        results = []
        for i in range(self.num_head):
            results.append(self.multi_head[i](x, mask))
        result = torch.cat(results, dim=1)
        y = self.linear(result)
        return y
