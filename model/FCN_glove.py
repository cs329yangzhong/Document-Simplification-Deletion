import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np
from torch.nn.parameter import Parameter
import math
import pickle



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


class FCN_Glove(nn.Module):

    def __init__(self, args,word_to_ix,  embedding_file, mlp_d=150,  encoding="utf-8", concat=False):
        super(FCN_Glove, self).__init__()
        self.args = args
        
        self.concat_num = 0
        D = 300
        if concat:
            self.concat_num = 200
            #final_D = D + self.concat_num
            final_D = 37 * (10+2)
        else:
             
            final_D = 37 
        C = 1
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        dropout_r = 0.1
        Ks = [int(y) for y in Ks.split(",")]
        self.embedding_dim = D
        self.voc_size = len(word_to_ix)
        mlp_d = final_D // 2
        self.binner = GaussianBinner()
        self.init_embedding(embedding_file, word_to_ix, encoding) 
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_r),
                                        nn.Linear(final_D, mlp_d),
                                        #nn.Tanh(),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_r),
                                        nn.Linear(mlp_d, mlp_d),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_r),
                                        nn.Linear(mlp_d,1),)
    
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

    def forward(self, x, pos, batch_size, l1=None):
        x = self.embeddings(x)  # (N, W, D)
        x = torch.mean(x, dim=1)
        conc_features = pos
        conc_features = conc_features.squeeze(1)
        sparse_features = conc_features.cpu().data.numpy()[:, :-200]
        surround_features = conc_features[:,200:]

       # assert(surround_features.shape[2] == 200 and sparse_features.shape[2] == 19)
       # if self.concat_num > 0:
        #    x = torch.cat((x, surround_features.squeeze(1)), 1)
        
        if self.concat_num > 0:
            self.binner.fit(sparse_features, sparse_features.shape[1])
            y = self.binner.transform(sparse_features, sparse_features.shape[1])
        else:
            y = sparse_features
        sparse_features = torch.FloatTensor(y).cuda()
        # (N,1,D)
        if self.args.use_gpu:
            x = x.cuda()
        logit = self.classifier(sparse_features)
        #logit = self.classifier(x)
         #   print(logit)
        return (logit.squeeze(1))
