from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch

from antispoof.base import BaseModel


class RNNwithBN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, batch_first=True):
        super(RNNwithBN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True, batch_first=batch_first)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x, output_lengths, h=None):
        if self.batch_norm is not None:
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        x = pack_padded_sequence(x, output_lengths, batch_first=True)
        x, h = self.rnn(x, h)
        x, output_lengths = pad_packed_sequence(x, batch_first=True)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x, h

class BatchRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.RNN, bidirectional=False, num_layers = 1, batch_first=True):
        super(BatchRNNLayer, self).__init__()
        self.rnn = nn.Sequential(
            RNNwithBN(
                input_size=input_size,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                batch_norm=False,
                batch_first=batch_first
            ),
            *(
                RNNwithBN(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    batch_first=batch_first
                ) for _ in range(num_layers - 1)
            )
        )

    def forward(self, input, output_lengths, h=None):
        x = input
        for i, rnn in enumerate(self.rnn):
            x, h = rnn(x, output_lengths, h=h)
        return x, h


class DeepSpeech2pacModel(BaseModel):
    def __init__(self, n_feats, n_class, hidden, conv_num, conv_type, 
                 kernel_sizes, strides, channels, paddings, rnn_type, 
                 rnn_layers, rnn_bidirectional, batch_norm_conv, **batch):
        super().__init__(n_feats, n_class, **batch)

        rnns = {
            'lstm': nn.LSTM,
            'rnn': nn.RNN,
            'gru': nn.GRU,
        }

        self.rnn_bidirectional = rnn_bidirectional
        self.kernel_sizes = kernel_sizes
        self.conv_num = conv_num
        self.strides = strides
        self.paddings = paddings
        self.channels = [1] + channels
        self.hidden = hidden
        self.n_class = n_class
        self.n_feats = n_feats

        self.out_conv_dim = n_feats
        for i in range(conv_num):
            self.out_conv_dim = ((self.out_conv_dim + 2*self.paddings[i][0] - self.kernel_sizes[i][0]) // self.strides[i][0]) + 1

        self.conv_layers = []

        for i in range(conv_num):
            if conv_type == '2d':
                self.conv_layers.append(nn.Conv2d(self.channels[i], self.channels[i+1], self.kernel_sizes[i], stride=self.strides[i], padding=self.paddings[i]))
            else:
                raise NotImplementedError
            
            if batch_norm_conv:
                self.conv_layers.append(nn.BatchNorm2d(self.channels[i+1]))

            self.conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.rnn = BatchRNNLayer(self.out_conv_dim * self.channels[-1], self.hidden, num_layers=rnn_layers, bidirectional=rnn_bidirectional, batch_first=True, rnn_type=nn.GRU)
        self.fc = nn.Linear(in_features=hidden, out_features=n_class)



    def forward(self, spectrogram, **batch):
        lengths = batch['spectrogram_length']
        output_lengths = self.transform_input_lengths(lengths)
        assert len(spectrogram.size()) == 3
        batch_size, dim, time = spectrogram.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_feats
        output = spectrogram.unsqueeze(1)

        output = self.conv_layers(output)

        batch_size, channels, dim, seq_l = output.size()
        assert channels == self.channels[-1]
        assert batch_size == len(batch['audio_path'])
        assert dim == self.out_conv_dim
        output = output.permute(0, 3, 1, 2)
        output = output.view(batch_size, seq_l, channels * dim)

        output, _ = self.rnn(output, output_lengths)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.hidden
        output = self.fc(output)
        batch_size, seq_l, dim = output.size()
        assert batch_size == len(batch['audio_path'])
        assert dim == self.n_class
        return {"logits": output}


    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for i in range(self.conv_num):
            output_lengths = ((output_lengths + 2*self.paddings[i][1] - self.kernel_sizes[i][1]) // self.strides[i][1]) + 1
        return output_lengths
