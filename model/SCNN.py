import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .layers import *

# Cell

class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'

def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x

class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super(ConvBlock, self).__init__()
        kernel_size = kernel_size
        self.layer_list = []

        self.conv = Conv1d_new_padding(ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero)
        self.bn = nn.BatchNorm1d(num_features=nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None: self.layer_list.append(act)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.net(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=39, bottleneck=True, pad_zero=True):
        super(InceptionModule, self).__init__()

        bottleneck = bottleneck if ni > 1 else False  ## first layer:False
        self.bottleneck = Conv1d_new_padding(ni, nf, 1, bias=False, pad_zero=pad_zero) if bottleneck else noop
        self.convs = Conv1d_new_padding(nf if bottleneck else ni, nf * (OUT_NUM), ks, bias=False, pad_zero=pad_zero)

        self.bn = nn.BatchNorm1d(nf * OUT_NUM)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.convs(x)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=47, depth=4, ks=39, pad_zero=True):
        super(InceptionBlock, self).__init__()
        self.depth = depth
        self.inception = nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * OUT_NUM, nf, ks=ks, pad_zero=pad_zero))

    def forward(self, x):
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
        return x


class SCNN_FC(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, pad_zero=True):
        super(SCNN_FC, self).__init__()
        self.inceptionblock = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * OUT_NUM, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SCNN(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, adaptive_size=50, pad_zero=False):
        super(SCNN, self).__init__()
        self.block = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.head_nf = nf * OUT_NUM
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, c_out, 1, act=None),
                                  GAP1d(1))

    def forward(self, x):
        x = self.block(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)


#########################增加部分
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, seq_len]
        batch, channels, _ = x.shape
        avg_pool = self.global_avg_pool(x).view(batch, channels)
        attention = self.fc2(torch.relu(self.fc1(avg_pool)))
        attention = self.sigmoid(attention).unsqueeze(2)
        return x * attention

# SCNN_Base：只负责特征提取，返回三维张量 [batch, channels, seq_len]
class SCNN_Base(nn.Module):
    def __init__(self, c_in, nf=47, depth=4, kernel=39, pad_zero=False):
        super(SCNN_Base, self).__init__()
        self.c_in = c_in  # 保存输入通道数
        self.block = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.head_nf = nf * OUT_NUM  # 提取后的特征通道数

    def forward(self, x):
        # 如果输入为二维 [batch, seq_len]，则重塑为 [batch, c_in, seq_len]
        if x.dim() == 2:
            x = x.view(x.size(0), self.c_in, -1)
        x = self.block(x)
        return x

# SCNN_Classifier：将特征映射到类别输出，输入应为 [batch, channels, seq_len]
class SCNN_Classifier(nn.Module):
    def __init__(self, c_out, in_channels, adaptive_size=50):
        super(SCNN_Classifier, self).__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_size),
            ConvBlock(in_channels, c_out, kernel_size=1, act=None),
            GAP1d(1)
        )

    def forward(self, x):
        x = self.head(x)  # 输出形状：[batch, c_out]
        return F.log_softmax(x, dim=1)

# SCNN_with_CA：结合特征提取、通道注意力和分类头
class SCNN_with_CA(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, pad_zero=False):
        super(SCNN_with_CA, self).__init__()
        self.feature_extractor = SCNN_Base(c_in, nf, depth, kernel, pad_zero)
        self.channel_att = ChannelAttention(in_channels=self.feature_extractor.head_nf)
        self.classifier = SCNN_Classifier(c_out, in_channels=self.feature_extractor.head_nf)

    def forward(self, x):
        # 提取特征，输出形状应为 [batch, channels, seq_len]
        x = self.feature_extractor(x)
        # 应用通道注意力
        x = self.channel_att(x)
        # 分类输出，形状为 [batch, c_out]
        x = self.classifier(x)
        return x

# 如果需要保留原始 SCNN_FC 和 SCNN 模型，也可如下定义：
class SCNN_FC(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, pad_zero=True):
        super(SCNN_FC, self).__init__()
        self.inceptionblock = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * OUT_NUM, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SCNN(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, adaptive_size=50, pad_zero=False):
        super(SCNN, self).__init__()
        self.block = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.head_nf = nf * OUT_NUM
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_size),
            ConvBlock(self.head_nf, c_out, kernel_size=1, act=None),
            GAP1d(1)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)