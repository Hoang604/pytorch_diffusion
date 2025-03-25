import torch 
import torch.nn as nn

x = torch.tensor([[[[1., 2., 3.],
                    [3., 2., 1.]],
                   [[4., 5., 6.],
                    [6., 5., 4.]]],
                   [[[7., 8., 9.],
                     [9., 8., 7.]],
                   [[10., 11., 12.],
                    [12., 11., 10.]]]])

print(x.shape)

print(x.transpose(0, 1))