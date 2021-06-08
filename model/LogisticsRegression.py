import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes, gpu=True):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.gpu = gpu

        if gpu:
            self.linear = self.linear.cuda()

    def forward(self, x):
        out = self.linear(x)

        out = F.sigmoid(out)

        if self.gpu == True:
            out = out.cuda()
        return out
