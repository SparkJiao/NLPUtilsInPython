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
        gate = torch.sigmoid(self.linear(torch.cat([y, i_x], dim=2)))
        output = y + i_x * gate
        return output


class IMNetwork(nn.Module):
    def __init__(self, input_dim):
        super(IMNetwork, self).__init__()
        self.im_layer = IMLayer(input_dim)

    def forward(self, x, x_mask):
        """
        :param x: b * seq *  m * h
        :param x_mask: b * seq * m
        :return: y: b * seq * m * h
        """
        x_t = x.transpose(0, 1)
        mask_t = x_mask.transpose(0, 1)
        y = x_t.new_zeros(x_t.size())
        y[0] = x_t[0]
        seq_len = x_t.size(0)
        for i in range(1, seq_len, 1):
            y[i] = self.im_layer(y[i - 1], x_t[i], mask_t[i - 1])
        result = y.transpose(0, 1)
        return result


class IMFusion(nn.Module):
    def __init__(self, input_dim):
        super(IMFusion, self).__init__()
        self.im_layer = IMLayer(input_dim)

    def forward(self, x, y):
        """
        :param x: b * seq * m * h
        :param y: b * seq * n * h
        :return: b * seq * n * h
        """
        x_t = x.transpose(0, 1)
        y_t = y.transpose(0, 1)
        z_t = y_t.new_zeros(y_t.size())
        seq_len = x_t.size(0)
        z_t[0] = y_t[0]
        for i in range(1, seq_len, 1):
            z_t[i] = self.im_layer(x_t[i - 1], y_t[i])
        return z_t.transpose(0, 1)
