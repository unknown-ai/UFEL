import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


### LeNet5 ###
class LeNet5(nn.Module):
    def __init__(self, z_dim=84, y_dim=10, w=32, h=32, c=3, rep=[1, 1, 1]):
        super(LeNet5, self).__init__()

        channels = [3, 64, 128]
        self.conv1 = nn.Conv2d(c, channels[1]*(1+rep[0]), kernel_size=5)
        self.act1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Conv2d(channels[1], channels[2]*(1+rep[1]), kernel_size=5)
        self.act2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(channels[2] * 5 * 5, 120)
        self.act3 = nn.Sequential(
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(120, z_dim)
        self.last_fc = nn.Linear(z_dim, y_dim*(1+rep[2]))
        self.w = w
        self.h = h
        self.c = c
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.rep = rep
        self.channels = channels

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def latent(self, x):
        x = self.conv1(x)
        if self.rep[0]:
            self.mu1 = x[:, :self.channels[1]]
            self.logvar1 = x[:, self.channels[1]:]
            x = self.reparameterize(self.mu1, self.logvar1)
        x = self.act1(x)
        x = self.conv2(x)
        if self.rep[1]:
            self.mu2 = x[:, :self.channels[2]]
            self.logvar2 = x[:, self.channels[2]:]
            x = self.reparameterize(self.mu2, self.logvar2)
        x = self.act2(x)
        x = x.view(-1, self.channels[2] * 5 * 5)
        x = self.fc1(x)
        x = self.act3(x)
        z = self.fc2(x)
        return z

    def forward(self, x):
        z = self.latent(x)
        z = self.act3(z)
        out = self.last_fc(z)
        if self.rep[2]:
            self.mu3 = out[:, :self.y_dim]
            self.logvar3 = out[:, self.y_dim:]
            out = self.reparameterize(self.mu3, self.logvar3)
        return out

### DenseNet ###
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, rep=[1, 1, 1]):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(
            math.floor(in_planes*reduction))*(1+rep[0]), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.logvar1_dim = in_planes
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(
            math.floor(in_planes*reduction))*(1+rep[1]), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        self.logvar2_dim = in_planes
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes*(1+rep[2]))
        self.in_planes = in_planes
        self.y_dim = num_classes
        self.rep = rep

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        if self.rep[0]:
            self.mu1 = out[:, :self.logvar1_dim]
            self.logvar1 = out[:, self.logvar1_dim:]
            out = self.reparameterize(self.mu1, self.logvar1)
        out = self.trans2(self.block2(out))
        if self.rep[1]:
            self.mu2 = out[:, :self.logvar2_dim]
            self.logvar2 = out[:, self.logvar2_dim:]
            out = self.reparameterize(self.mu2, self.logvar2)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        if self.rep[2]:
            self.mu3 = out[:, :self.y_dim]
            self.logvar3 = out[:, self.y_dim:]
            out = self.reparameterize(self.mu3, self.logvar3)
        return out

### WideResNet ###

class BasicBlock_resnet(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock_resnet, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, rep=[1, 1, 1]):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock_resnet
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1]*(1+rep[0]), block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2]*(1+rep[1]), block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes*(1+rep[2]))
        self.nChannels = nChannels
        self.num_classes = num_classes
        self.rep = rep

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_() * 10 ** (-30) # scaling
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        if self.rep[0]:
            self.mu1 = out[:, :self.nChannels[1]]
            self.logvar1 = out[:, self.nChannels[1]:]
            out = self.reparameterize(self.mu1, self.logvar1)
        out = self.block2(out)
        if self.rep[1]:
            self.mu2 = out[:, :self.nChannels[2]]
            self.logvar2 = out[:, self.nChannels[2]:]
            out = self.reparameterize(self.mu2, self.logvar2)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])

        out = self.fc(out)
        if self.rep[2]:
            self.mu3 = out[:, :self.num_classes]
            self.logvar3 = out[:, self.num_classes:]
            out = self.reparameterize(self.mu3, self.logvar3)

        return out

### feature combining CNN ###

class Feature_Ensemble_CNN_for_Lenet(nn.Module):
    def __init__(self, y_dim):
        super(Feature_Ensemble_CNN_for_Lenet, self).__init__()

        channels = [128, 64, 1]
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=1, padding=2) 
        self.act1 = nn.Sequential(
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=1, padding=2) 
        self.act2 = nn.Sequential(
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(channels[2] * 3 * 3 + y_dim + 1, 100)
        self.act3 = nn.Sequential(
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(100, 1)
        self.channels = channels


    def forward(self, x_conv, x_fc, x_softmax):
        x = self.conv1(x_conv)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.view(-1, self.channels[2] * 3 * 3)
        x = torch.cat([x, x_fc, x_softmax], dim=1)        
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        return x
    
class Feature_Ensemble_CNN_for_Dense(nn.Module):
    def __init__(self, y_dim=100):
        super(Feature_Ensemble_CNN_for_Dense, self).__init__()

        channels = [108, 150, 64, 1]
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=1, padding=2) # 16 → 8
        self.act1 = nn.Sequential(
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Conv2d(channels[1]*2, channels[2], kernel_size=4, stride=1,padding=2) # 8 → 4
        self.act2 = nn.Sequential(
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=1,padding=2) # 4 → 2
        self.act3 = nn.Sequential(
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )        
        self.fc1 = nn.Linear(channels[3] * 2 * 2 + y_dim + 1, 1024)
        self.act4 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(1024, 1)
        self.channels = channels


    def forward(self, x_conv1, x_conv2, x_fc, x_softmax):
        x = self.conv1(x_conv1)
        x = self.act1(x)
        x = torch.cat([x, x_conv2], dim=1)             
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)  
        x = x.view(-1, self.channels[3] * 2 * 2)
        x = torch.cat([x, x_fc, x_softmax], dim=1)        
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x
    
class Feature_Ensemble_CNN_for_resnet(nn.Module):
    def __init__(self, y_dim=100):
        super(Feature_Ensemble_CNN_for_resnet, self).__init__()

        channels = [64, 128, 64, 1]
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=1, padding=2) # 32 → 16
        self.act1 = nn.Sequential(
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Conv2d(channels[1]*2, channels[2], kernel_size=4, stride=1, padding=2) # 16 → 8
        self.act2 = nn.Sequential(
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=1, padding=2) # 8 → 4
        self.act3 = nn.Sequential(
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )        
        self.fc1 = nn.Linear(channels[3] * 4 * 4 + y_dim + 1, 1024)
        self.act4 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(1024, 1)
        self.channels = channels


    def forward(self, x_conv1, x_conv2, x_fc, x_softmax):
        x = self.conv1(x_conv1)
        x = self.act1(x)
        x = torch.cat([x, x_conv2], dim=1)             
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)        
        x = x.view(-1, self.channels[3] * 4 * 4)
        x = torch.cat([x, x_fc, x_softmax], dim=1)        
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x    
