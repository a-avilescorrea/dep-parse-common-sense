import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size=1024, hidden_size=10, output_size=1, dropout=0.0001):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, embedded_input):
        embedded_input = embedded_input.to(device)
        h0 = torch.zeros(1, 1, self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(1, 1, self.hidden_size).requires_grad_().to(device)
        output, (hidden_state, cell_state) = self.rnn(embedded_input, (h0.detach(), c0.detach()))
        hidden_state = hidden_state.squeeze()
        return self.linear(hidden_state)

    def predict(self, sentence):
        logits = self.forward(sentence)
        return torch.argmax(self.softmax(logits))


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'lstm.th'))


def load_model():
    from torch import load
    from os import path
    r = LSTM()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'lstm.th'), map_location='cpu'))
    return r
