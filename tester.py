import torch
import torch.nn as nn
import json
import argparse

from allennlp.modules.elmo import Elmo, batch_to_ids


class WinoGrandeSentence(object):

    def __init__(self, sentence_1, sentence_2, answer):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.answer = answer

    def __repr__(self):
        return str(self.sentence_1, self.sentence_2, self.answer)

    def __str__(self):
        return self.__repr__()

    def get(self):
        return self.sentence_1, self.sentence_2, self.answer


def create_data(file_path):
    data_list = list()

    f = open(file_path, 'r')
    for l in f:
        line = json.loads(l)
        wino_sent = WinoGrandeSentence(line['sentence_1'], line['sentence_2'], line['answer'])
        data_list.append(wino_sent)

    return data_list

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


def get_embeddings(sentence, elmo):

    sentences = [sentence.split()]
    character_ids = batch_to_ids(sentences)

    return elmo(character_ids)


def load_model():
    from torch import load
    from os import path
    r = LSTM()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'lstm.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='dev_parsed.jsonl')
    args = parser.parse_args()

    model = load_model()
    model = model.to(device)
    model.eval()
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    data = create_data(args.test_set)

    total_correct = 0
    for l in data:
        sentence_1, sentence_2, label = l.get()
        label = int(label)
        embedding = get_embeddings(sentence_1, elmo)
        prediction = torch.round(torch.sigmoid(model(embedding['elmo_representations'][0]))).to('cpu').type(torch.LongTensor)
        prediction = prediction.item()
        if prediction is 1 and label is 1:
            total_correct += 1

        embedding = get_embeddings(sentence_2, elmo)
        prediction = torch.round(torch.sigmoid(model(embedding['elmo_representations'][0]))).to('cpu').type(torch.LongTensor)
        prediction = prediction.item()
        if prediction is 1 and label is 2:
            total_correct += 1

    accuracy = total_correct * 1.0 / len(data)

    print("Guessed %i out of %i" % (total_correct, len(data)))
    print("accuracy = %.3f" % accuracy)

