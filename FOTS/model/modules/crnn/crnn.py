import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths)
        recurrent, _ = self.rnn(packed_input)  # [T, b, h * 2]
        padded_input, actual_length = torch.nn.utils.rnn.pad_packed_sequence(recurrent)

        T, b, h = padded_input.size()
        t_rec = padded_input.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output, actual_length


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

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
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
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output, actual_length = self.rnn(conv, lengths)

        return output, actual_length