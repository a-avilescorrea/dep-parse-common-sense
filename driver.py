import argparse
import json

import torch
import torch.nn as nn
import numpy as np
import torch.utils.tensorboard as tb

import random
from allennlp.modules.elmo import Elmo, batch_to_ids

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_embeddings(sentence, elmo):

    sentences = [sentence.split()]
    character_ids = batch_to_ids(sentences)

    return elmo(character_ids)


def get_wg_data(args):
    raw_data = open(args.dataset, 'r')

    sentences = list()
    labels = list()
    for l in raw_data:
        line_json = json.loads(l)
        sentence = line_json['sentence']
        # preprocess
        sentence = sentence.lower()
        sentence_1 = sentence.replace('_', line_json['option1'].lower())
        sentence_2 = sentence.replace('_', line_json['option2'].lower())
        label = line_json['answer']
        sentences.append((sentence_1, sentence_2))
        labels.append(label)
    return sentences, labels


def train(args, data):
    from os import path
    lr = args.lr
    epochs = args.epochs
    hidden_size = 10
    dropout = 0.0001
    num_classes = 1
    input_size = 1024

    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=num_classes, dropout=dropout).to(device)

    # create Elmo object
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss = nn.BCEWithLogitsLoss()

    if args.logger is not None:
        train_logger = tb.SummaryWriter(path.join(args.logger, 'train_%f' % lr), flush_secs=1)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0

        train_idx = [i for i in range(len(data))]
        random.shuffle(train_idx)
        for i in train_idx:
            example_loss = 0.0
            wg = data[i]
            sentence_1, sentence_2, label = wg.get()
            labels = [[1], [0]] if label == 1 else [[0], [1]]
            labels = torch.FloatTensor(labels).to(device)
            embedding = get_embeddings(sentence_1, elmo)
            logit1 = model(embedding['elmo_representations'][0])
            loss_val = loss(logit1, labels[0])
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss_val.item()
            example_loss += loss_val.item()
            global_step += 1

            embedding = get_embeddings(sentence_2, elmo)
            logit2 = model(embedding['elmo_representations'][0])
            loss_val = loss(logit2, labels[1])
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss_val.item()
            example_loss += loss_val.item()
            global_step += 1
            if train_logger is not None:
                train_logger.add_scalar('loss', example_loss, global_step)

        print("loss on epoch:%i: %f" % (epoch, total_loss))
        save_model(model, lr)
    model.eval()
    save_model(model, lr)


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

    def __init__(self, input_size=1024, hidden_size=10, output_size=1, dropout=0.001):
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


def save_model(model, lr):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'lstm_%f.th' % lr))


def load_model():
    from torch import load
    from os import path
    r = LSTM()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'lstm.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WinoGrande Challenge. By Alberto Aviles-Correa. ')

    #Required
    parser.add_argument('--dataset', type=str, help='path to dataset',default='winogrande_1.1/train_s.jsonl')
    parser.add_argument('--dependency', type=bool, help='set true if using dependency parsed sentences', default=True)
    parser.add_argument('--logger')
    parser.add_argument('--continue_training')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    data = create_data(args.dataset)
    train(args, data)




