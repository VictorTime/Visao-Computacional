import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            # # TODO   
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# class ConvLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ConvLayer, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, input):
#         return self.conv(input)

# class D_ConvLayer(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(D_ConvLayer, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch, 3, padding=1),
#             nn.BatchNorm2d(in_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, input):
#         return self.conv(input)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        """
        F_g : nº de canais do sinal do decoder
        F_x : nº de canais do skip connection (encoder)
        F_int: nº de canais intermediários (redução dimensional)
        """
        super(AttentionGate, self).__init__()
        
        # Decoder
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        # Skip
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        # Combine
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: feature map do decoder (usado como 'gating signal')
        x: skip connection vindo do encoder
        """
        g1 = self.W_g(g)     # B, F_int, H, W
        x1 = self.W_x(x)     # B, F_int, H, W
        psi = self.relu(g1 + x1)
        psi = self.psi(psi) 
        out = x * psi
        return out
    
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, H*W)          # B, C//8, HW
        proj_key   = self.key_conv(x).view(B, -1, H*W)            # B, C//8, HW
        proj_value = self.value_conv(x).view(B, -1, H*W)          # B, C, HW

        # Similaridade
        energy = torch.bmm(proj_query.permute(0,2,1), proj_key)   # B, HW, HW
        attention = F.softmax(energy, dim=-1)

        out = torch.bmm(proj_value, attention.permute(0,2,1))     # B, C, HW
        out = out.view(B, C, H, W)

        # Residual
        out = self.gamma * out + x
        return out


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # self.attn_t3 = SelfAttention2D(kan_input_dim)
        # self.attn_t2 = SelfAttention2D(kan_input_dim//4)
        # self.attn_t1 = SelfAttention2D(kan_input_dim//8)
        
        F_g = embed_dims[2]      # gating com 512 canais (ajuste conforme necessário)
        F_x_t3 = kan_input_dim   # canais de t3 = 256 (exemplo)
        F_x_t2 = kan_input_dim//4  # canais de t2 = 256//4 = 64
        F_x_t1 = kan_input_dim//8  # canais de t1 = 256//8 = 32

        self.att_t3 = AttentionGate(F_g=F_g, F_x=F_x_t3, F_int=F_x_t3//2)
        self.att_t2 = AttentionGate(F_g=F_g, F_x=F_x_t2, F_int=F_x_t2//2)
        self.att_t1 = AttentionGate(F_g=F_g, F_x=F_x_t1, F_int=F_x_t1//2)
        
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        g = out  # Gating signal (poderia ser após o primeiro decoder step, se preferir)

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))

        out = torch.add(out, t4)
        B, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        g_for_t3 = F.interpolate(g, size=t3.shape[2:], mode='bilinear', align_corners=False)
        t3_att = self.att_t3(g_for_t3, t3)
        # t3 = self.att_t3(t3)
        out = torch.add(out,t3_att)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        # Bloco para Self Attention
        # for i, blk in enumerate(self.dblock2):
        #     out = blk(out, H, W)

        # out = self.dnorm4(out)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # t2 = self.att_t2(t2)
        # out = torch.add(out,t2)
        # out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        # t1 = self.att_t1(t1)
        # out = torch.add(out,t1)
        # out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        
        for blk in self.dblock2:
            out = blk(out, t3_att.shape[2], t3_att.shape[3])

        out = self.dnorm4(out)
        out = out.reshape(B, t3_att.shape[2], t3_att.shape[3], -1).permute(0, 3, 1, 2).contiguous()
        
        ### DECODER 3 (256 -> 64)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear'))
        g_for_t2 = F.interpolate(g, size=t2.shape[2:], mode='bilinear', align_corners=False)
        t2_att = self.att_t2(g_for_t2, t2) # t2 tem 64 canais
        out = torch.add(out, t2_att)

        ### DECODER 4 (64 -> 32)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear'))
        g_for_t1 = F.interpolate(g, size=t1.shape[2:], mode='bilinear', align_corners=False)
        t1_att = self.att_t1(g_for_t1, t1) # t1 tem 32 canais
        out = torch.add(out, t1_att)

        ### DECODER 5 (32 -> 32)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear'))
        return self.final(out)