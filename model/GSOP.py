# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:40:54 2024

@author: ZY
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.MPNCOV import CovpoolLayer

def cov_feature(x):
    # 计算协方差矩阵
    # Input: x.shape = (B,C,8,8)
    batchsize = x.data.shape[0]  # B
    dim = x.data.shape[1]        # C
    h = x.data.shape[2]          # 8
    w = x.data.shape[3]          # 8
    M = h * w                    # M = 64
    
    # x: (B,C,8,8) -> (B,C,64)
    x = x.reshape(batchsize, dim, M)
    
    # I_hat: (C,C) -> (1,C,C) -> (B,C,C)
    I_hat = (-1. / M / M) * torch.ones(dim, dim, device=x.device) + (1. / M) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    
    # x.transpose(1,2): (B,C,64) -> (B,64,C)
    # bmm(I_hat): (B,64,C) x (B,C,C) -> (B,64,C)
    # bmm(x): (B,64,C) x (B,C,64) -> (B,64,64)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    
    # Output: y.shape = (B,64,64)
    return y


class GSoP(nn.Module):

    def __init__(self, indim, attention='0', att_dim=128):
        super(GSoP, self).__init__()
        self.dimDR = att_dim   # 128
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        if attention in {'1', '+', 'M', '&'}:
            # if planes > 64:
            #     DR_stride=1
            # else:
            #     DR_stride=2
            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(indim, self.ch_dim, kernel_size=1, stride=1, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            # row-wise conv is realized by group conv
            self.row_conv_group = nn.Conv2d(self.ch_dim, 4 * self.ch_dim, kernel_size=(self.ch_dim, 1),
                                            groups=self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(4 * self.ch_dim, indim, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()

        if attention in {'2', '+', 'M', '&'}:
            self.sp_d = att_dim    # 128
            self.sp_h = 8
            self.sp_w = 8
            self.sp_reso = self.sp_h * self.sp_w  # 64
            self.conv_for_DR_spatial = nn.Conv2d(indim, self.sp_d, kernel_size=1, stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h, self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            # row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d(self.sp_reso, self.sp_reso * 4, kernel_size=(self.sp_reso, 1),
                                                        groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Conv2d(self.sp_reso * 4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

        if attention == '&':  # we employ a weighted spatial concat to keep dim
            self.groups_base = 32
            self.groups = int(indim / 64)
            self.factor = int(math.log(self.groups_base / self.groups, 2))
            self.padding_num = self.factor + 2
            self.conv_kernel_size = self.factor * 2 + 5
            self.dilate_conv_for_concat1 = nn.Conv2d(indim,indim, kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.dilate_conv_for_concat2 = nn.Conv2d(indim, indim, kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.bn_for_concat = nn.BatchNorm2d(indim)

        self.attention = attention

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)  # down channel
        out = self.bn_for_DR(out)
        out = self.relu(out)  # NxCxHxW

        out = CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        return out

    def pos_att(self, out):
        # 1. 特征预处理
        pre_att = out  # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)  # NxCxHxW -> Nxsp_dxHxW
        out = self.bn_for_DR_spatial(out)  # 维度不变

        # 2. 自适应池化到固定大小
        out = self.adppool(out)  # Nxsp_dxHxW -> Nxsp_dx8x8
        # 例如: [64, 128, 7, 7] -> [64, 128, 8, 8]

        # 3. 计算协方差特征
        out = cov_feature(out)  # 计算协方差矩阵

        # 4. 特征转换
        out = out.view(out.size(0), out.size(1), out.size(2), 1)  # N x sp_dxsp_d -> Nx sp_d x sp_d x1
        out = self.row_bn_for_spatial(out)  # 维度不变
        out = self.row_conv_group_for_spatial(out)  # N x sp_d x sp_d x1 -> N x (sp_reso*4) x1x1
        out = self.relu(out)  # 维度不变

        # 5. 生成注意力图
        out = self.fc_adapt_channels_for_spatial(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w)

        # 6. 上采样到原始大小
        out = self.adpunpool(out, (pre_att.size(2), pre_att.size(3)))

        return out

    def forward(self, x):

        out = x  # x.shape = (B,C,H,W)
        if self.attention == '1':  # channel attention,GSoP default mode
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        elif self.attention == '2':  # position attention
            pre_att = out
            att = self.pos_att(out)
            out = self.relu_normal(pre_att * att)

        elif self.attention == '+':  # fusion manner: average
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = pre_att * chan_att + self.relu(pre_att.clone() * pos_att)

        elif self.attention == 'M':  # fusion manner: MAX
            pre_att = out  # 64 768 7 7
            chan_att = self.chan_att(out)  # 64 768 1 1
            pos_att = self.pos_att(out)     # 64 1 7 7
            out = torch.max(pre_att * chan_att, self.relu(pre_att.clone() * pos_att))  # 64 768 7 7 all

        elif self.attention == '&':  # fusion manner: concat
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out1 = self.dilate_conv_for_concat1(pre_att * chan_att)
            out2 = self.dilate_conv_for_concat2(self.relu(pre_att * pos_att))
            out = out1 + out2
            out = self.bn_for_concat(out)

        return out.contiguous()
