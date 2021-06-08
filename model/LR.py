import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np
from torch.nn.parameter import Parameter
import math
import pickle


class LR(nn.Module):

    def __init__(self, args,word_to_ix, embedding_file,embed=True, encoding="utf-8"):
        super(LR, self).__init__()
        self.args = args
        self.add_features = args.add_features

        self.conc = 219
        D = 300
        self.embedding_dim = 300
        self.voc_size = len(word_to_ix)

        self.init_embedding(embedding_file, word_to_ix, encoding)
        self.embed = embed
        self.conc = args.conc
        print(self.conc)
        # classifier layer.
        if embed and not self.add_features:
            self.linear = nn.Linear(D, 1)
            print("embedding only")
        elif embed and self.add_features:
            self.linear = nn.Linear(D + self.conc, 1)
            print("emedding all features")

        else:
            self.linear = nn.Linear(self.conc, 1)
       #self.hidden_dim = args.hidden_dim
       # self.label = nn.Linear(len(Ks)*Co+self.fc_dim, 1)

        # no additional features.
        

    def init_embedding(self, embedding_file, word_to_ix, encoding):
        print("start embeddings___")
        word_embedding_dict = {}
        # read the embedding file and store to a dictionary {keys = words, values = numpy array of weights }
        try:
            embedding_matrix = pickle.load(open("embeddings.pkl",'rb'))
            print("load prestored embedding metrics")

        except FileNotFoundError:
            with open(embedding_file, encoding=encoding) as f:
                for line in f:
                    line = line.strip().split()
                    word_embedding_dict[line[0].lower()] = np.asarray(line[1:])

            # initialize the embedding weights
            # size of (Nums of vocabularies from the corpus + 2, embedding dimensions)
            embedding_matrix = np.zeros((self.voc_size + 2, self.embedding_dim))

            for idx, word in enumerate(word_to_ix.keys()):
                if word in word_embedding_dict.keys():
                    embedding_matrix[idx] = word_embedding_dict[word]
                else:
                    embedding_matrix[idx] = np.random.random(self.embedding_dim) * -2 + 1
            pickle.dump(embedding_matrix, open('embeddings.pkl', 'wb'))

        # copy weights to enbedding layer
        self.embeddings = nn.Embedding(len(embedding_matrix), self.embedding_dim)
        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embeddings.weight.requires_grad = False

        logging.info('Word_Embedding initialized')
        logging.info('Num of Embedding = {0} '
                     'Embedding dim = {1}'.format(len(embedding_matrix), self.embedding_dim))
    def forward(self, x, pos, batch_size, l1=None):
        x = self.embeddings(x)  # (N, W, D)
        x = torch.mean(x, dim=1)
        pos = pos.squeeze(1)
        #print(x.shape, pos.shape)
        if self.embed and self.add_features:
            out = self.linear(torch.cat((x, pos), 1))
        elif self.embed and not self.add_features:
            out = self.linear(x)
        return out.squeeze(1)
