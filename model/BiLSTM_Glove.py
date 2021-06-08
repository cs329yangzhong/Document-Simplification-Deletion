import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np
from torch.nn.parameter import Parameter
import math
import pickle


## Gaussian Binner. 

def gaussian(diff, sig):
    return np.exp(-np.power(diff, 2.) / (2 * sig * sig))

class GaussianBinner:

    def __init__(self, bins=10, w=0.1):
        self.bin_values, self.sigmas = [], []
        self.bins = bins
        self.width = w
        self.eps = 0.000001

    def fit(self, x, features_to_be_binned):
        for index in range(0, features_to_be_binned):
            dimension = x[:, index]
            bin_divisions = np.histogram(dimension, bins=self.bins)[1]

            bin_means = [(bin_divisions[i] + bin_divisions[i + 1]) / 2.0
                         for i in range(0, len(bin_divisions) - 1)]

            half_width = abs(bin_divisions[1] - bin_divisions[0]) / 2.0
            bin_means[0:0] = [bin_divisions[0] - half_width]
            bin_means.append(bin_divisions[len(bin_divisions) - 1] + half_width)
            self.bin_values.append(bin_means)

            self.sigmas.append(abs(bin_divisions[1] - bin_divisions[0]) * self.width)

    def transform(self, x, features_to_be_binned):
        expanded_features = [x[:, features_to_be_binned:]]
        for index in range(0, features_to_be_binned):
            bin_means = np.array(self.bin_values[index])

            projected_features = gaussian(np.tile(x[:, index], (self.bins + 2, 1)).T - bin_means,
                                          self.sigmas[index])

            sum_f = np.sum(projected_features, axis=1)
            sum_f[sum_f == 0] = self.eps
            projected_features = (projected_features.T / sum_f).T
            expanded_features.append(projected_features)

        return np.concatenate(expanded_features, axis=1)


class BiLSTM_glove(nn.Module):

    def __init__(self, args,word_to_ix, embedding_file,Gaussian_num=10,  conc = False, encoding="utf-8", num_layers = 1):
        super(BiLSTM_glove, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.use_gpu = args.use_gpu
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lstm = nn.LSTM(input_size = args.embed_dim,
                            hidden_size = args.hidden_dim,
                            dropout = self.dropout,
                            bidirectional = True,
                            num_layers = num_layers)


        conc_features = 37
        self.conc_features = 37      
        self.binner_num = Gaussian_num
        dropout_r = args.dropout   
        if self.binner_num != None:
            self.binned_features = self.conc_features * (self.binner_num+2)
        else:
            self.binned_features = self.conc_features

        self.use_gpu = True
        self.num_layer = num_layers
        self.Dropout = nn.Dropout(self.dropout)
        self.hidden = self.init_hidden(self.batch_size)

        self.embedding_dim = 300
        self.voc_size = len(word_to_ix)
        self.init_embedding(embedding_file, word_to_ix, encoding)
       
        self.label_noSF = nn.Linear(args.hidden_dim*2,1)

        self.embedding_dim = 300
        if Gaussian_num != None:
            self.binner = GaussianBinner(bins=Gaussian_num, w=0.2)

        mlp_d = self.binned_features // 2
        self.inputdim2 = args.hidden_dim*2 + mlp_d
        final_mlp_d = self.inputdim2 // 2 
        self.fc_2 = nn.Sequential(
                                        nn.Linear(self.binned_features, mlp_d),
                                       nn.ReLU(),
                                        nn.Dropout(p=dropout_r),
                                       nn.Linear(mlp_d, mlp_d),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_r),)
        
        self.label = nn.Sequential(nn.Linear(self.inputdim2, final_mlp_d),
                                   nn.BatchNorm1d(final_mlp_d),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_r),
                                   nn.Linear(final_mlp_d, final_mlp_d),
                                   nn.BatchNorm1d(final_mlp_d),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_r),
                                   nn.Linear(final_mlp_d, 1),
                                   )
        
        if not conc:
            self.conc_features = 0

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
    
    def init_hidden(self, batch_size):
        if self.use_gpu:
            return (Variable(torch.zeros(2*self.num_layer, batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2*self.num_layer, batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2*self.num_layer, batch_size, self.hidden_dim)),
            Variable(torch.zeros(2*self.num_layer, batch_size, self.hidden_dim)))
        print("initial hidden states")

    def forward(self, x, conc_features,  batch_size, l1=None):
        x = self.embeddings(x)  # (N, W, D)
        if self.args.use_gpu:
            x = x.cuda()
        x = x.permute(1,0,2)  
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(x, hidden)
       

        sparse_features = conc_features.cpu().data.numpy()[:,:, :-200]
        surround_features = conc_features[:,:,-200:]

        assert(surround_features.shape[2] == 200 and sparse_features.shape[2] == 37)
        sparse_features = np.squeeze(sparse_features, axis=1)
        if self.binner_num != None:
            self.binner.fit(sparse_features, sparse_features.shape[1])
            y = self.binner.transform(sparse_features, sparse_features.shape[1])
        else:
            y = sparse_features
        sparse_features = torch.FloatTensor(y)
        if self.use_gpu:
            sparse_features = sparse_features.cuda()
        
        learned_y = self.fc_2(sparse_features)

        if self.conc_features == 0:
            y = self.label_noSF(lstm_out[-1])
            y = torch.squeeze(y)
            return y
            
        else:
            y = self.label(torch.cat((lstm_out[-1], learned_y),1))
            return y.squeeze()
