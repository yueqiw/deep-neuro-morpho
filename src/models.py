import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import sparseconvnet as scn


# two-dimensional SparseConvNet
class VGGplus1(nn.Module):
    def __init__(self, num_classes):
        nn.Module.__init__(self)
        self.sparseModel = scn.SparseVggNet(2, 1, [
            ['C', 8, ], ['C', 8], 'MP',
            ['C', 16], ['C', 16], 'MP',
            ['C', 16, 8], ['C', 16, 8], 'MP',
            ['C', 24, 8], ['C', 24, 8], 'MP']
            ).add(scn.Convolution(2, 32, 64, 5, 1, False)
            ).add(scn.BatchNormReLU(64)
            ).add(scn.SparseToDense(2,64))
        self.linear = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.sparseModel(x)
        x = x.view(-1,64)
        x = self.linear(x)
        return x
