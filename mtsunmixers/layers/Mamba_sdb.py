import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.Selective_scan_interface import selective_scan_fn
import torch.nn.functional as F
import math

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device='cuda'):
        '''
        The improvement of layer norm
        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class S6(nn.Module):
    def __init__(self, d_model, d_state=16, d_inner=128, dt_rank="auto", dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4, use_scan_cuda=True):
        super(S6, self).__init__()

        self.d_model = d_model  # N token特征维度
        self.d_state = d_state  # D 隐藏状态数
        self.d_inner = d_inner
        self.use_scan_cuda = use_scan_cuda
        # self.d_conv = d_conv  # kernel 1D卷积宽度，默认为4
        # self.expand = expand  # 特征维扩大因子，默认为2
        # self.d_inner = int(self.expand * self.d_model)  # 扩增后特征维度，即通道数
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 内部秩大小

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)  # 投影生成Delta， B和C  2N->N/16+D+D

        dt_init_std = self.dt_rank ** -0.5 * dt_scale  # 1/sqrt(rank)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)  # 常数初始化
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)  # 均匀分布
        else:
            raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            # dt_min为1e-3,dt_max为0.1,
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  ### clamp函数最小值为1e-4
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))  ### 求逆
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)  ## 梯度不更新
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()  # Hippo矩阵A
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def forward(self, u):
        batch, dim, seqlen = u.shape

        A = -torch.exp(self.A_log.float())

        # assert self.activation in ["silu", "swish"]
        x_dbl = self.x_proj(rearrange(u, "b d l -> (b l) d"))  # (bl d)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj.weight @ delta.t()
        delta = rearrange(delta, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        delta_bias = self.dt_proj.bias.float()
        delta_softplus = True
        dtype_in = u.dtype
        if not self.use_scan_cuda:
            u = u.float()
            delta = delta.float()
            if delta_bias is not None:
                delta = delta + delta_bias[..., None].float()
            if delta_softplus:
                delta = F.softplus(delta)  # delta = log(1+exp(delta))
            batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
            is_variable_B = B.dim() >= 3
            is_variable_C = C.dim() >= 3
            if A.is_complex():
                if is_variable_B:
                    B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
                if is_variable_C:
                    C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
            else:
                B = B.float()
                C = C.float()
            x = A.new_zeros((batch, dim, dstate))
            ys = []
            deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
            if not is_variable_B:
                deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
            else:
                if B.dim() == 3:
                    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
                else:
                    B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            if is_variable_C and C.dim() == 4:
                C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            last_state = None
            for i in range(u.shape[2]):
                x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
                if not is_variable_C:
                    y = torch.einsum('bdn,dn->bd', x, C)
                else:
                    if C.dim() == 3:
                        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                    else:
                        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
                if i == u.shape[2] - 1:
                    last_state = x
                if y.is_complex():
                    y = y.real * 2
                ys.append(y)
            y = torch.stack(ys, dim=2)  # (batch dim L)
            out = y
            out = out.to(dtype=dtype_in)
            return out

        else:
            y = selective_scan_fn(
                u,
                delta,
                A,
                B,
                C,
                None,
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            out = y
            out = out.to(dtype=dtype_in)
            return out


class MambaBlock(nn.Module):
    def __init__(self, d_model, state_size, expand=2, d_conv=3, conv_bias=True, use_casual1D=True, bias=False):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = state_size
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.d_inner = int(self.expand * self.d_model)

        self.norm = RMSNorm(self.d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1
        )  # 2N->2N, 逐通道卷积, L+d_conv-1
        self.activation = "silu"  # 激活函数 y=x*sigmoid(x)
        self.act = nn.SiLU()
        self.use_casual1D = use_casual1D
        self.ssm = S6(d_model=self.d_model, d_state=self.d_state, d_inner=self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)  ###最后将输出维度还原

    def forward(self, x):
        # Reference to Fig.3 in the original paper of Mamba
        ### 0.9996687685 one layer, no gate
        pre_x = x
        x = self.norm(x)  # RMSNorm, an improvement for Layer norm  [b,l,d]

        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=x.shape[1],
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)

        assert self.activation in ["silu", "swish"]
        if self.use_casual1D:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation
            )
        else:
            x = self.act(self.conv1d(x)[..., :x.shape[-1]])
        x_ssm = self.ssm(x)  # The SSM to capture the long-range dependencies

        x_residual = self.act(z)  # The linear representation and followed a Silu activation to obtain the gated key
        x_combined = x_ssm * x_residual  # Key and value are multiplied
        x_combined = rearrange(x_combined, "b d l -> b l d")
        x_out = self.out_proj(x_combined)  # Adjust the channel dimension to the initial ones
        return x_out + pre_x


class Mamba(nn.Module):
    def __init__(self, d_model, state_size, layer=1):
        super(Mamba, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(1, layer + 1):
            self.layers.append(
                MambaBlock(d_model=d_model, state_size=state_size, expand=2, d_conv=3, bias=False)
            )

    def forward(self, x):
        for ma in self.layers:
            x = ma(x)
        return x


if __name__ == '__main__':
    import time

    start = time.perf_counter()
    mamba = Mamba(64, 16, 3).cuda()
    x = torch.randn(80, 10000, 64).cuda()
    y = mamba(x)
    print(y.shape)
    end = time.perf_counter()
    print('time is %s' % (end - start))
