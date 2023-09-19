import math
import torch.nn as nn
import torch.autograd.variable
from thop import profile, clever_format
from torch.autograd import Variable
import pdb

class AttnBlock(nn.Module):
    def __init__(self, feature_dim, infer=False, causal=True):
        super(AttnBlock, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.query_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        self.key_vector = nn.Parameter(torch.FloatTensor(self.feature_dim, ), requires_grad=True)
        self.value_vector = nn.Parameter(torch.FloatTensor(self.feature_dim), requires_grad=True)
        self.query_linear = nn.Linear(self.feature_dim, self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.value_linear_sig = nn.Linear(self.feature_dim, self.feature_dim)
        self.value_linear_tan = nn.Linear(self.feature_dim, self.feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.query_vector.size(0))
        self.query_vector.data.uniform_(-stdv, stdv)
        self.key_vector.data.uniform_(-stdv, stdv)
        self.value_vector.data.uniform_(-stdv, stdv)

    def forward(self, query, value):
        query = self.query_linear(query)
        query = self.query_norm(query)
        q_vector = self.sigmoid(self.query_vector)
        query = query * q_vector

        k_vector = self.sigmoid(self.key_vector)
        key = value * k_vector

        v_vector = self.sigmoid(self.value_vector)
        if not self.infer:
            v_sigm = self.sigmoid(self.value_linear_sig(v_vector))
            v_tanh = self.tanh(self.value_linear_tan(v_vector))
            v_vector = v_sigm * v_tanh
        value = value * v_vector

        weight = query.matmul(key.transpose(1, 2) / math.sqrt(self.feature_dim))
        if self.causal:
            # torch.ones_like会继承目标的梯度，是否有影响？
            mask = (1 - torch.tril(torch.ones_like(weight))).type(torch.bool)
            weight = weight.masked_fill(mask, float('-inf'))
            weight = self.softmax(weight)
        else:
            weight = self.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0])

        out = weight.matmul(value)
        return out



class FeedForward(nn.Module):
    def __init__(self, feature_dim,channel ):
        super(FeedForward, self).__init__()
        self.feature_dim = feature_dim
        self.input_to_half =nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=(1,1))

        self.input_proj = nn.Linear(self.feature_dim, 4 * self.feature_dim)
        self.gussian_elu = nn.GELU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, input):
        half_out = self.input_to_half(input)
        out = self.input_proj(half_out)
        out = self.gussian_elu(out)
        out = self.dropout(out)
        out = torch.split(out, self.feature_dim, -1)
        return out[0] + out[1] + out[2] + out[3]+half_out


class ArnBlock(nn.Module):
    def __init__(self, feature_dim,channel, infer=False, causal=True):
        super(ArnBlock, self).__init__()
        self.feature_dim = feature_dim
        self.infer = infer
        self.causal = causal
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.rnn = nn.LSTM(input_size=self.feature_dim,
                           hidden_size=self.feature_dim if causal else (self.feature_dim // 2), num_layers=1,
                           batch_first=True,
                           bidirectional=(not self.causal))
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = AttnBlock(self.feature_dim, self.infer, self.causal)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = FeedForward(self.feature_dim,channel=channel)

    def forward(self, input):
        self.rnn.flatten_parameters()
        out = self.input_norm(input)
        out, _ = self.rnn(out)
        value = self.value_norm(out)
        query = self.query_norm(out)
        out = self.attention(query, value)
        out = query + out
        feed_in = self.feed_norm(out)
        out = self.out_norm(out)
        feed_out = self.feedforward(feed_in)
        out = out + feed_out

        return out
