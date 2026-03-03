import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
class RecurrentModel(nn.Module):

    def __init__(self, model_name):
        super(RecurrentModel, self).__init__()
        self.model_name = model_name

        model_ins = nn.RNN
        if model_name == "lstm":
            model_ins = nn.LSTM
        elif model_name == "gru":
            model_ins = nn.GRU

        self.model = model_ins(768, 32, 2, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):

        x, x_lengths = batch.x, batch.x_lengths
        x_pack = pack_sequence(x, x_lengths)
        output, hidden = self.model(x_pack)

        seq_unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)

        res = self.linear(seq_unpacked.mean(1))
        res = self.softmax(res)

        return res
