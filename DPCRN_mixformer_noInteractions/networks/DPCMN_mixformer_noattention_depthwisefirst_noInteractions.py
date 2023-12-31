# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
from networks.nnstruct.arnblock import *


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm([c, f])

    def forward(self, x):
        x = self.ln(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(32, 32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        # self.conv_1 = nn.Conv2d(18, 32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_1 = nn.PReLU(32)

        self.conv_2 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_3 = nn.PReLU(32)

        self.conv_4 = nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = nn.Conv2d(64, 80, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_5 = nn.BatchNorm2d(80, eps=1e-8)
        self.act_5 = nn.PReLU(80)

    def forward(self, x):
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:, :, :-1, :]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:, :, :-1, :]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:, :, :-1, :]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:, :, :-1, :]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:, :, :-1, :]))
        return [x_1, x_2, x_3, x_4, x_5]


class Decoder(nn.Module):
    def __init__(self, auto_encoder=True):
        super(Decoder, self).__init__()

        self.imag_dconv_1 = nn.ConvTranspose2d(80 * 2, 64, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)

        self.imag_dconv_2 = nn.ConvTranspose2d(128, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_2 = nn.PReLU(32)

        self.imag_dconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)

        self.imag_dconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 2))
        self.imag_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_4 = nn.PReLU(32)

        self.imag_dconv_5 = nn.ConvTranspose2d(64, 2, kernel_size=(2, 5), stride=(1, 2))
        self.imag_bn_5 = nn.BatchNorm2d(2, eps=1e-8)
        self.imag_act_5 = nn.PReLU(2)

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4], dprnn_out], 1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:, :, :-1, 1:-1]))
        skipcon_2 = torch.cat([encoder_out[3], x_1], 1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:, :, :-1, 1:-1]))
        skipcon_3 = torch.cat([encoder_out[2], x_2], 1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:, :, :-1, 1:-1]))
        skipcon_4 = torch.cat([encoder_out[1], x_3], 1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:, :, :-1, :-1]))
        skipcon_5 = torch.cat([encoder_out[0], x_4], 1)
        x_5 = self.imag_dconv_5(skipcon_5)[:, :, :-1, 1:-1]

        return x_5


class Interaction(nn.Module):
    def __init__(self, channel):
        super(Interaction, self).__init__()
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(channel, eps=1e-8)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.gussian_elu = nn.GELU()

    def forward(self, input):
        out = self.conv_1(input)
        out = self.bn_1(out)
        out = self.gussian_elu(out)
        out = self.conv_2(out)
        out = self.sigmoid(out)
        return out


class DPRNN(nn.Module):
    def __init__(self, numUnits=80, width=50, channel=80, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=int(self.numUnits // 2), batch_first=True,
                                 bidirectional=True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm([50, channel])
        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True,
                                 bidirectional=False)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm([50, channel])

        self.width = width
        self.channel = channel

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
            ## Intra RNN
        # import pdb; pdb.set_trace()
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0]  # (Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1],
                                                 inter_LSTM_input.shape[2], inter_LSTM_input.shape[3])  # (Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        inter_out = self.inter_ln(inter_ln_input)

        #  inter_out = inter_out.permute(0,3,1,2) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)
        inter_out = inter_out.contiguous()

        return inter_out


class Mixformer(nn.Module):
    def __init__(self, numUnits=80, width=50, channel=80, **kwargs):
        super(Mixformer, self).__init__(**kwargs)
        # mixformer
        self.feature_dim = width
        self.channel = channel
        self.width = width
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.depthwise = nn.Conv2d(self.channel, self.channel, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2),
                                   groups=self.channel)
        self.value_norm = nn.LayerNorm(self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.attention = AttnBlock(self.feature_dim)
        self.feed_norm = nn.LayerNorm(self.feature_dim)
        self.out_norm = nn.LayerNorm(self.feature_dim)
        self.deepwise_norm = nn.LayerNorm(self.feature_dim)
        self.feedforward = FeedForward(self.feature_dim, channel=channel * 2)
        # self.Channel_Interaction = Interaction(self.channel)
        # self.Spatial_Interaction = Interaction(self.channel)

        self.DPRNN1 = DPRNN(80, 50, 80)
        self.DPRNN2 = DPRNN(80, 50, 80)
        self.attention_conv = AttnBlock(self.feature_dim)

    def forward(self, x):
        # B = x.size(0)
        # mix_conv
        intra_depthwise_tomix = self.depthwise(x)
        intra_depthwise_tomix = self.DPRNN1(intra_depthwise_tomix)

        # intra_depthwise_tomix_toattn = self.Channel_Interaction(intra_depthwise_tomix)  # B,C,F,T
        ## Inter RNN

        inter_out = x
        inter_out = self.DPRNN2(inter_out)

        inter_out = inter_out.contiguous()
        inter_out = inter_out.view(inter_out.shape[0] * inter_out.shape[1],
                                   inter_out.shape[2], inter_out.shape[3])
        inter_out = self.input_norm(inter_out)

        # intra_depthwise = intra_depthwise_tomix_toattn.contiguous()
        # intra_depthwise = intra_depthwise.view(intra_depthwise.shape[0] * intra_depthwise.shape[1],
        #                                        intra_depthwise.shape[2], intra_depthwise.shape[3])

        value = self.value_norm(inter_out)
        query = self.query_norm(inter_out)
        # out = self.attention(query, value * intra_depthwise)

        out = self.attention(query, value)
        out = query + out

        out = out.contiguous()
        inter_out_mixout = out.view(x.shape[0], self.channel, -1, self.width)
        out = self.out_norm(inter_out_mixout)
        # inter_out_mixout_todepth = self.Spatial_Interaction(inter_out_mixout)
        # intra_depthwise_mixout = intra_depthwise_tomix * inter_out_mixout_todepth
        intra_depthwise_mixout = intra_depthwise_tomix
        intra_depthwise_mixout = self.deepwise_norm(intra_depthwise_mixout)
        #  inter_out = inter_out.permute(0,3,1,2) #(Bs, T, F, C)
        inter_out = torch.cat([intra_depthwise_mixout, out], 1)
        feed_in = self.feed_norm(inter_out)
        feed_out = self.feedforward(feed_in)
        out = out + intra_depthwise_mixout + feed_out
        out = out.contiguous()
        return out


class DPCMN_mixformer(nn.Module):
    # autoencoder = True
    def __init__(self):
        super(DPCMN_mixformer, self).__init__()
        self.encoder = Encoder()
        self.mixformer1 = Mixformer(80, 50, 80)
        self.mixformer2 = Mixformer(80, 50, 80)
        self.decoder = Decoder()
        self.n_fft = 400
        self.hop_length = 200

    def forward(self, x):
        # x : b, mic, wave

        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, center=True)
        # X: bm F T ri
        F, T = X.shape[1], X.shape[2]
        X = X.reshape(b, m, F, T, 2)
        X0 = torch.cat([X[..., 0], X[..., 1]], dim=1)
        # x:[batch, channel, frequency, time]

        X0 = X0.permute(0, 1, 3, 2)  # (Bs, c, T, F)
        encoder_out = self.encoder(X0)

        dprnn_out = self.mixformer1(encoder_out[4])
        dprnn_out = self.mixformer2(dprnn_out)

        Y = self.decoder(dprnn_out, encoder_out)
        Y = Y.permute(0, 1, 3, 2)
        #  import pdb; pdb.set_trace()

        m_out = int(Y.shape[1] // 2)
        Y_r = Y[:, :m_out]
        Y_i = Y[:, m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)

        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, center=True, length=t)
        y = y.reshape(b, m_out, y.shape[-1])

        return y


def complexity():
    from ptflops import get_model_complexity_info
    model = DPCMN_mixformer()
    mac, param = get_model_complexity_info(model, (16, 16000), as_strings=True, print_per_layer_stat=True, verbose=True)
    print(mac, param)


if __name__ == '__main__':
    complexity()
