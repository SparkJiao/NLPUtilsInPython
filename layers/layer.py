import torch
from torch import nn

from layers.attention import MultiHeadSelfAtt


class IMLayer(nn.Module):
    def __init__(self, input_dim):
        super(IMLayer, self).__init__()
        self.multi_head_self = MultiHeadSelfAtt(input_dim, input_dim)
        self.linear = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x, y):
        """
        IM Layer
        :param x: information to integrate in y: b * m * h
        :param y: information to wait fusing: b * n * h
        :return: r: b * n * h
        """
        # b * h
        self_x = self.multi_head_self(x)
        # b * n * h
        i_x = self_x.unsqueeze(1).repeat(1, y.size(1), 1)
        gate = torch.sigmoid(self.linear(torch.cat([y, i_x]), dim=2))
        output = y + i_x * gate
        return output
