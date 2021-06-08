import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np
from torch.nn.parameter import Parameter
import math
import pickle
from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import batch_to_ids

class ElmoCNN(nn.Module):

    def __init__(self, args, encoding="utf-8"):
        super(ElmoCNN, self).__init__()
        self.args = args

        D = 300
        C = 1
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        Ks = [int(y) for y in Ks.split(",")]
        self.embedding_dim = D

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

        self.elmo = Elmo(options_file, weight_file, 1, dropout=args.dropout,
                         do_layer_norm=False)

        self.conc = 0
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, 512)) for K in Ks])
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
                                            nn.ReLU(),
                                            nn.Dropout(p=args.dropout),
                                            nn.Linear(self.fc_dim,
                                                      self.fc_dim), )

            self.label = nn.Linear(len(Ks) * Co + self.fc_dim, 1)
        else:

            self.fc1 = nn.Linear(len(Ks) * Co + self.conc, C)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, features, batch_size, l1=None):
        x = batch_to_ids(x)

        if self.args.use_gpu:
            x = x.cuda()
        elmo_out = self.elmo(x)
        x = elmo_out['elmo_representations'][0]  # (N, W, D)

        if self.args.use_gpu:
            x = x.cuda()
        print(x.shape)
        x = x.permute(1,0,2)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in
             x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)i

        if self.conc == 0:
            logit = self.fc1(x)
            return logit.squeeze(1)

        else:

            # Learn embedding from features.
            learned_embed = self.classifier(features.squeeze(1))
            conc_result = torch.cat((x, learned_embed), 1)
            logit = self.label(conc_result)
            return (logit.squeeze(1))
