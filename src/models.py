import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import sparseconvnet as scn


# two-dimensional SparseConvNet
class SparseResNet2D(nn.Module):
    def __init__(self, n_classes):
        nn.Module.__init__(self)
        self.n_classes = n_classes 
        self.sparseModel = scn.Sequential(
            # 255x255
            scn.SubmanifoldConvolution(2, 2, 16, 3, False),  # dimension, nIn, nOut, filter_size, bias
            scn.MaxPooling(2, 3, 2),  # dimension, pool_size, pool_stride
            # 127x127
            scn.SparseResNet(2, 16, [  # dimension, nInputPlanes, layers
                        ['b', 16, 2, 1],  # 63x63  # blockType, n, reps, stride
                        ['b', 32, 2, 2],  # 63x63
                        ['b', 48, 2, 2],  # 31x31
                        ['b', 96, 2, 2],  # 15x15 
                        ['b', 144, 2, 2],  # 7x7
                        ['b', 192, 2, 2]]),  # 3x3
            scn.Convolution(2, 192, 256, 3, 1, False),  # 1x1 # dimension, nIn, nOut, filter_size, filter_stride, bias
            scn.BatchNormReLU(256))  # dimension, nPlanes
        self.sparse_to_dense = scn.SparseToDense(2, 256)
        #self.spatial_size= self.sc.input_spatial_size(torch.LongTensor([1, 1]))
        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        self.inputLayer = scn.InputLayer(2, self.spatial_size, 2)  # dimension, spatial_size, mode
        self.linear = nn.Linear(256, self.n_classes)
        print(self.spatial_size)

    def forward(self, x):
        x = self.inputLayer(x)
        # print(x.spatial_size) 
        x = self.sparseModel(x)
        x = self.sparse_to_dense(x)
        # print(x.size())
        x = x.view(-1, 256)
        x = self.linear(x)
        return x


