__all__ = ['MTS_UNMixers']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from layers.RevIN import RevIN
from layers.Mamba_dual import Mamba_bi
from layers.Mamba_sdb import Mamba

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = MTS_UNMixers(config)

    def forward(self, x, *args, **kwargs):
        x = x.permute(0, 2, 1)
        x, outputs,  channel_output_res,time_output_res_o = self.model(x)
        outputs = outputs.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        outputs_channel = channel_output_res.permute(0, 2, 1)
        outputs_time = time_output_res_o.permute(0, 2, 1)
        return x, outputs, outputs_channel, outputs_time


class MTS_UNMixers(nn.Module):
    def __init__(self,
                 config, **kwargs):
        super().__init__()
        self.configs = config
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.d_model1 = config.d_model1
        patch_num = int((config.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        k_time = config.k_time
        k = config.k

        # patch mamba
        self.mamba_patch = Mamba(d_model=self.patch_num, state_size=32)#nn.Linear(self.patch_num, self.patch_num)#
        self.Linear_time_patch = nn.Linear(self.d_model1 * self.patch_num, k_time)

        # channel mamba
        self.mamba_channel = Mamba_bi(d_model=self.d_model, state_size=32) # nn.Linear(self.d_model, self.d_model)  # ###############no mamba
        self.Linear_channel = nn.Linear(self.d_model, k)

        #BiMambaEncoder(d_model, n_state)
        self.mlp_channel = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.d_model))

        # E
        self.E_time_patch = nn.Parameter(torch.randn(k_time, config.seq_len))
        self.E_time_patch_fore = nn.Parameter(torch.randn(k_time, config.pred_len))
        self.E_channel = nn.Parameter(torch.randn(k, config.seq_len))
        self.E_channel_fore = nn.Parameter(torch.randn(k, config.pred_len))

        # bia
        self.bias = nn.Parameter(torch.randn(config.enc_in, config.seq_len))
        self.bias1 = nn.Parameter(torch.randn(config.enc_in, config.pred_len))
        self.E_time_patch1 = torch.nn.Parameter(torch.abs(torch.randn(k_time, config.seq_len)))  # Endmembers
        self.E_channel1 = torch.nn.Parameter(torch.abs(torch.randn(k, config.seq_len)))  # Abundance

        # head
        self.W_z_out = nn.Linear(config.pred_len + config.pred_len, config.pred_len)
        self.dropout_head = nn.Dropout(config.head_dropout)

        self.Linear_d_model = nn.Linear(config.seq_len, self.d_model)
        self.Linear_d_model1 = nn.Linear(self.patch_len, self.d_model1)

        # head structure
        self.W_restructure = nn.Linear(config.seq_len + config.seq_len, config.seq_len)
        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)


    def forward(self, z, *args, **kwargs):
        # norm
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)

        # channel decomposition & time feature
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [b,v,n,p]
        zcube = self.Linear_d_model1(zcube)   # [b,v,n,d_model1]
        zcube = zcube.permute(0, 1, 3, 2)  # [b,v,d_model1,n]
        z_time_mamba = torch.reshape(zcube, (zcube.shape[0] * zcube.shape[1], zcube.shape[2], zcube.shape[3]))  # [b*v,d_model1,n]
        for _ in range(self.configs.n_layers):
            z_time_mamba = self.mamba_patch(z_time_mamba)
        z_time_mamba = torch.reshape(z_time_mamba, (-1, zcube.shape[1], zcube.shape[2], zcube.shape[3]))  # [b,v,d_model1,n]
        z_time_A = torch.reshape(z_time_mamba, (z_time_mamba.shape[0], z_time_mamba.shape[1], z_time_mamba.shape[2] * z_time_mamba.shape[3]))  # [b,v,d_model1*n]
        z_time_A = self.Linear_time_patch(z_time_A)  # (b,v,k_time)
        z_time_A = F.softmax(z_time_A, dim=-1)
        E_time_patch = self.E_time_patch  # (k_time,l)
        E_time_patch_fore = self.E_time_patch_fore  # (k_time,t)
        time_output_res = torch.matmul(z_time_A, E_time_patch)  # + self.bias # (b,v,l)
        z_time = torch.matmul(z_time_A, E_time_patch_fore)  # + self.bias1  # (b,v,t)

        # channel feature & time decomposition
        z_channel_input = z  # [b,v,l]
        z_channel_input = self.Linear_d_model(z_channel_input)
        z_channel_mamba = self.mamba_channel(z_channel_input)
        z_channel_A = self.Linear_channel(z_channel_mamba)
        z_channel = z_channel_A#torch.relu(z_channel_A)  # z_channel_A#

        E_channel = self.E_channel
        E_channel = F.softmax(E_channel, dim=1)
        E_channel_fore = self.E_channel_fore
        E_channel_fore = F.softmax(E_channel_fore, dim=1)
        channel_output_res = torch.matmul(z_channel, E_channel)   # (b,v,l)
        z_channel = torch.matmul(z_channel, E_channel_fore)  # (b,v,t)

        # z_out  output
        z_out = torch.cat((z_channel, z_time), dim=-1)
        z_out = self.W_z_out(z_out)#x_channel_out + x_time_out.permute(0, 2, 1)
        z_out = self.dropout_head(z_out)
        z_out = z_out.permute(0, 2, 1)
        z_out = self.revin_layer(z_out, 'denorm')
        z = z_out.permute(0, 2, 1)

        # restructure
        time_output_res_o = time_output_res
        time_output_res = torch.cat((time_output_res, channel_output_res), dim=-1)
        time_output_res = self.W_restructure(time_output_res)

        # denorm
        time_output_res = time_output_res.permute(0, 2, 1)
        time_output_res = self.revin_layer(time_output_res, 'denorm')
        time_output_res = time_output_res.permute(0, 2, 1)

        time_output_res_o = time_output_res_o.permute(0, 2, 1)
        time_output_res_o = self.revin_layer(time_output_res_o, 'denorm')
        time_output_res_o = time_output_res_o.permute(0, 2, 1)

        channel_output_res = channel_output_res.permute(0, 2, 1)
        channel_output_res = self.revin_layer(channel_output_res, 'denorm')
        channel_output_res = channel_output_res.permute(0, 2, 1)

        return z, time_output_res, channel_output_res,time_output_res_o

