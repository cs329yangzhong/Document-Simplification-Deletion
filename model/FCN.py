import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
import logging
import torch_util

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



class FCN(nn.Module):
    
    
    def __init__(self, args,word_to_ix, embedding_file, encoding='utf-8',mlp_d_embed=150,
                    mlp_d=150, batch_norm=True, 
                    dropout_r=0.1, max_l=60,concat=True, Gaussian_num=None):
        
        super(FCN, self).__init__()
        
        self.max_l = 60
        self.conc = 0
        self.conc_features = 3    
        self.binner_num = Gaussian_num
       
        if self.binner_num != None:
            self.binned_features = self.conc_features * (self.binner_num+2)
        else:
            self.binned_features = self.conc_features

        self.use_gpu = True

        D = 300
        if concat:
            self.concat_num = 0
            final_D = D + self.concat_num
        else:
            self.concat_num = 0
            final_D = D 
        
        dropout_r = 0.5
        self.embedding_dim = D
        self.voc_size = len(word_to_ix)

        self.init_embedding(embedding_file, word_to_ix, encoding) 
        mlp_d_embed = int(0.5 * final_D)
        self.embed_classifier = nn.Sequential(
                                        nn.Linear(final_D, mlp_d_embed),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_r),
                                        nn.Linear(mlp_d_embed, mlp_d_embed),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_r),
                                        nn.Linear(mlp_d_embed ,mlp_d_embed),)
        if Gaussian_num != None:
            self.binner = GaussianBinner(bins=Gaussian_num, w=0.2)
        print(self.binned_features)
        
        self.inputdim2 = mlp_d_embed + self.binned_features
        print(self.inputdim2)
        # raise EOFError
        final_mlp_d = self.inputdim2 // 2 
        # if self.concat_num == 0:
        #     self.inputdim2 = mlp_d_embed 

        if batch_norm:
            
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
        else:
            
            self.label = nn.Sequential(nn.Linear(self.inputdim2, final_mlp_d),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_r),
                                   nn.Linear(final_mlp_d, final_mlp_d),
                                   nn.ReLU(),
                                   nn.Dropout(p=dropout_r),
                                   nn.Linear(final_mlp_d, 1),
                                   )
                    
        
    def display(self):
        for param in self.parameters():
            print(param.data.size())
    
    def init_embedding(self, embedding_file, word_to_ix, encoding):
        print("start embeddings___")
        word_embedding_dict = {}
        # read the embedding file and store to a dictionary 
        # { words, weights }
        try:
            embedding_matrix = pickle.load(open("embeddings.pkl",'rb'))
            print("load prestored embedding metrics")

        except FileNotFoundError:
            with open(embedding_file, encoding=encoding) as f:
                for line in f:
                    line = line.strip().split()
                    word_embedding_dict[line[0].lower()] = np.asarray(line[1:])

            # initialize the embedding weights
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
        self.embeddings.weight.requires_grad = True

        logging.info('Word_Embedding initialized')
        logging.info('Num of Embedding = {0} '
                     'Embedding dim = {1}'.format(len(embedding_matrix), self.embedding_dim))
        print("finish word_embedding")
        
        
    def forward(self, sent, conc_features, l1, batch_size):
        sparse_features = conc_features.cpu().data.numpy()[:,:, -self.conc_features:]
        # surround_features = conc_features[:,:,-200:]
        # print(sparse_features.shape)
        # print(self.concat_num)
        # raise EOFError
        
        x = self.embeddings(sent)
        x = torch.mean(x, dim=1)

        # assert(surround_features.shape[2] == 200 and sparse_features.shape[2] == 37)
        sparse_features = np.squeeze(sparse_features, axis=1)
        if self.binner_num != None:
            self.binner.fit(sparse_features, sparse_features.shape[1])
            y = self.binner.transform(sparse_features, sparse_features.shape[1])
        else:
            y = sparse_features
        sparse_features = torch.FloatTensor(y)
        if self.use_gpu:
            sparse_features = sparse_features.cuda()
        
        # first feed the sentence embedding.
        x = self.embeddings(sent)
        x = torch.mean(x, dim=1)
        # surround_features = surround_features.squeeze(1)
        
        learned_x = self.embed_classifier(x)
        #learned_y = self.feature_classifier((sparse_features))
        ## Concatenate output of embedding with the sparse features.
     
        learned_y = sparse_features
        features = torch.cat((learned_x, learned_y),1) if learned_y.shape[-1] != 0 else learned_x 
        # print(features.shape)
        #out = self.bn2(out)
        out = self.label(features)
        # print(out)
        return out.squeeze() 
