
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import numpy as np

class CNN_Sent(nn.Module):

    def __init__(self, args):
        super(CNN_Sent, self).__init__()
        self.args = args

        V = 128
        D = args.embed_dim
        C = args.numClass
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        Ks = [int(y) for y in Ks.split(",")]

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co + 0, C)
        self.conc = args.conc

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, pos):
        x = torch.FloatTensor(x) # (N, W, D)
        if self.args.use_gpu:
            x = x.cuda()

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)i
        #pos = torch.FloatTensor(pos).cuda()
        #print(pos.shape)
        #print(x.shape)
        if self.conc == 0:
            logit = self.fc1(x)
            return logit.squeeze(1)
        logit = self.fc1(torch.cat((x, pos),1))
        #print(logit.squeeze(1))
        return (logit.squeeze(1))
