import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, lengths):
        self.rnn.flatten_parameters()
        total_length = input.size(1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        recurrent, _ = self.rnn(packed_input)  # [T, b, h * 2]
        padded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(recurrent, total_length=total_length, batch_first=True)

        b, T, h = padded_input.size()
        t_rec = padded_input.contiguous().view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)
        output = nn.functional.log_softmax(output, dim=-1) # required by pytorch's ctcloss

        return output


class HeightMaxPool(nn.Module):

    def __init__(self, size=(2, 1), stride=(2, 1)):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=size, stride=stride)

    def forward(self, input):
        return self.pooling(input)


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256]

        cnn = nn.Sequential()

        def convRelu(i):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        cnn.add_module('HeightMaxPooling{0}'.format(0), HeightMaxPool())
        convRelu(2)
        convRelu(3)
        cnn.add_module('HeightMaxPooling{0}'.format(1), HeightMaxPool())
        convRelu(4)
        convRelu(5)
        cnn.add_module('HeightMaxPooling{0}'.format(2), HeightMaxPool())

        self.cnn = cnn
        self.rnn = BidirectionalLSTM(256, nh, nclass)

    def forward(self, input, lengths):
        # conv features
        conv = self.cnn(input)

        # b, c, h, w_after = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # _, _, _, w_before = input.size()
        # step = (w_before / w_after).ceil()
        # padded_width_after = (lengths - 1 / step).ceil()

        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # [B, T, C]

        # rnn features
        output = self.rnn(conv, lengths)

        return output
