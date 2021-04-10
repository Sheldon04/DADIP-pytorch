import torch
import torch.nn as nn
from .common import *

'''
def fcn(num_input_channels=200, num_output_channels=1, num_hidden=800):


    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
    model.add(nn.ReLU6())

    model.add(nn.Linear(num_hidden, num_output_channels))
    #model.add(nn.ReLU6())
    model.add(nn.Softmax())

    return model
'''

'''
class Predictor(nn.Module):
    def __init__(self, in_nc=1, nf=64, size=27 * 27, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=11, stride=1, padding=2, bias=False),
            #nn.LeakyReLU(0.2, True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            #nn.LeakyReLU(0.2, True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
            #nn.LeakyReLU(0.2, True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, nf*2, kernel_size=5, stride=1, padding=2, bias=use_bias),
            #nn.LeakyReLU(0.2, True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf*2, nf*4, kernel_size=5, stride=1, padding=2, bias=use_bias),
            #nn.LeakyReLU(0.2, True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf*4, size, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.Softmax(),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, a, b):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(-1, 1, a, b)  # torch size: [B, code_len]
        

'''

'''

class Predictor(nn.Module):
    def __init__(self, in_nc=1, nf=128, size=27 * 27, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=11, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, size, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Softmax(),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, a, b):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(-1, 1, a, b)  # torch size: [B, code_len]
'''
class SEBlock(nn.Module):
    def __init__(self, channel, r=16):#16
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class Predictor(nn.Module):
    def __init__(self, in_nc=1, nf=128, size=27 * 27, use_bias=False):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=11, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, size, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            #nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(size, size, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Softmax(),
        ])
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))
        #self.seBlock = SEBlock(size)

    def forward(self, input, a, b):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        #re = self.seBlock(flat)
        return flat.view(-1, 1, a, b)  # torch size: [B, code_len]

