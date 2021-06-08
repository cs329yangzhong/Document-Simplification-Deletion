import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.autograd import Variable
import numpy as np

class BiLSTM_Classifier(nn.Module):
    def __init__(self, args):
        """

        """
        super(BiLSTM_Classifier, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.use_gpu = args.use_gpu
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.lstm = nn.LSTM(input_size=args.embed_dim, hidden_size=args.hidden_dim, dropout=self.dropout, bidirectional=True)
        conc_features = 0
        self.hidden2Label = nn.Linear(args.hidden_dim*2 + (conc_features), args.numClass)

        self.Dropout = nn.Dropout(self.dropout)
        self.hidden = self.init_hidden(args)
        self.conc = args.conc
        self.conc_feature = conc_features

    def init_hidden(self, args):
        if args.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))
        print("initialed")

    def forward(self, sentenceVec, pos):
        batch_size, seq_len, embd_dim = sentenceVec.shape
        x = torch.FloatTensor(sentenceVec.reshape((seq_len, batch_size,-1))) # reshape the input embeddding.

        if self.use_gpu:
            x = x.cuda()

        embedded = self.Dropout(x)
        lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        #print(lstm_out[-1].shape)
        if self.conc == 0:
            y = self.hidden2Label(lstm_out[-1])
            return y
        pos = torch.FloatTensor(pos).cuda()
        #print(pos.unsqueeze(1).shape)
        y = self.hidden2Label(torch.cat((lstm_out[-1], pos),1))
        #output = torch.sigmoid(y)
        return y

