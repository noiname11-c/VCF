
import torch
import torch.nn as nn
import torch.nn.functional as F
#
# 优化说明：
#   1. 修复 forward_gating 双重激活 Bug：原代码在 forward() 中对 g/v 先调用
#      act_gate，再将已激活的张量传入 forward_gating，而 forward_gating 内部
#      又再次调用 act_gate，导致激活被施加两次；现改为直接传入原始 g/v；
#   2. 新增 ChannelAttention（SE 模块）：在 value 分支后加入轻量通道注意力，
#      自适应地对多阶卷积输出的各通道重要性进行校准，提升判别力；
#   3. 新增 Pre-Norm（GroupNorm）：在 proj_1 之前对输入做归一化，使特征
#      分解更稳定，缓解深层网络的训练不稳定问题。

# Spatial Block with Multi-order Gated Aggregation
def build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAttention(nn.Module):
    """轻量 SE（Squeeze-and-Excitation）通道注意力模块。

    在 value 分支后对各通道重要性进行校准：
        1. Global Average Pooling 压缩空间维度；
        2. 两层全连接（bottleneck）学习通道间关系；
        3. Sigmoid 生成通道权重，乘回原特征图。
    """
    def __init__(self, embed_dims, reduction=4):
        super(ChannelAttention, self).__init__()
        mid_dims = max(embed_dims // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dims, mid_dims, bias=False),
            nn.GELU(),
            nn.Linear(mid_dims, embed_dims, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w


class MultiOrderDWConv(nn.Module):

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

class MultiOrderGatedAggregation(nn.Module):

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32

        # Pre-Norm：在 proj_1 之前做归一化，稳定特征分解过程
        self.pre_norm = nn.GroupNorm(1, embed_dims)

        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # SE 通道注意力：对 value 分支输出的各通道重要性做校准
        self.channel_attn = ChannelAttention(embed_dims, reduction=4)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        """fp32 精度下的门控聚合（修复：不再对已激活张量重复激活）。"""
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()

        # Pre-Norm + 特征分解
        x = self.pre_norm(x)
        x = self.feat_decompose(x)

        # 门控分支 & 值分支
        g = self.gate(x)
        v = self.value(x)

        # SE 通道注意力校准 value
        v = self.channel_attn(v)

        # 门控聚合
        # 修复：原代码在 attn_force_fp32=True 时将已激活的 g/v 传入 forward_gating，
        # 导致 act_gate 被施加两次；现统一传入原始 g/v，由各分支负责激活。
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(g, v)  # ← 修复：传原始 g, v

        x = x + shortcut
        return x


if __name__ == '__main__':
    input = torch.randn(1, 64, 32, 32)# 输入 B C H W
    block = MultiOrderGatedAggregation(embed_dims=64)
    output = block(input)
    print(input.size())
    print(output.size())
