import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np
from torch.nn.parameter import Parameter
import math
import pickle


class CNN_Glove(nn.Module):

    def __init__(self, args,word_to_ix, embedding_file, encoding="utf-8"):
        super(CNN_Glove, self).__init__()
        self.args = args

        D = 300
        C = 1
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        Ks = [int(y) for y in Ks.split(",")]
        self.embedding_dim = D
        self.voc_size = len(word_to_ix)

        self.init_embedding(embedding_file, word_to_ix, encoding)
        self.conc = 0
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.fc_dim = 100
        self.dropout = nn.Dropout(args.dropout)
        self.non_linear = args.non_linear
        # classifier layer.
        print("have conc ", self.conc)
        if self.non_linear and self.conc != 0:
            self.classifier = nn.Sequential(nn.Dropout(p=args.dropout),
                                        nn.Linear(self.conc, self.fc_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=args.dropout),
                                        nn.Linear(self.fc_dim, self.fc_dim),
                                        nn.ReLU(),                                                                                                   nn.Dropout(p=args.dropout),                                                                                  nn.Linear(self.fc_dim, self.fc_dim),)
            self.label = nn.Linear(len(Ks)*Co+self.fc_dim, 1)
        else:
        
            self.fc1 = nn.Linear(len(Ks)*Co+ self.conc, C)

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
        print("finish word_embedding")
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, pos, batch_size, l1=None):
        x = self.embeddings(x)  # (N, W, D)
        if self.args.use_gpu:
            x = x.cuda()
        #print(x.shape)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        #print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)i
        #self.conc = 0
        
        if self.conc == 0:
            
            logit = self.fc1(x)
            return logit.squeeze(1)

        else:
            #print("use additional feature")
            learned_embed = self.classifier(pos.squeeze(1))

            conc_result = torch.cat((x, learned_embed), 1)
        #    print(conc_result.shape)
            logit = self.label(conc_result)
         #   print(logit)
            return (logit.squeeze(1))
