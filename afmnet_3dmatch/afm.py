import torch
import torch.nn as nn
import torch.nn.functional as F
from afmnet_3dmatch.utils import grid_sample_wrapper, softmax, timer
from afmnet_3dmatch.mlp import Conv1dNormRelu, Conv2dNormRelu


class AFM(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn='af', norm=None):
        super().__init__()

        self.mlps3d = Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == 'concat':
            self.fuse2d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm)
            self.fuse3d = ConcatFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm)
        elif fusion_fn == 'af':
            self.fuse2d = AFFusion(in_channels_2d, in_channels_3d, in_channels_2d, 'nchw', norm, reduction=2)
            self.fuse3d = AFFusion(in_channels_2d, in_channels_3d, in_channels_3d, 'ncm', norm, reduction=2)
        else:
            raise ValueError

    @timer.timer_func
    def forward(self, uv, feat_2d, feat_3d):
        feat_2d = feat_2d.float()
        feat_3d = feat_3d.float()

        feat_2d_sampled = grid_sample_wrapper(feat_2d.detach(), uv)
        feat_2d_sampled_norm = F.normalize(feat_2d_sampled.squeeze(0).permute(1, 0), p=2, dim=1)
        out3d = self.fuse3d(self.mlps3d(feat_2d_sampled.detach()), feat_3d)

        return out3d, feat_2d_sampled_norm


class ConcatFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None):
        super().__init__()

        if feat_format == 'nchw':
            self.mlp = Conv2dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        elif feat_format == 'ncm':
            self.mlp = Conv1dNormRelu(in_channels_2d + in_channels_3d, out_channels, norm=norm)
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        return self.mlp(torch.cat([feat_2d, feat_3d], dim=1))


class AFFusion(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None, reduction=1):
        super().__init__()

        if feat_format == 'nchw':
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif feat_format == 'ncm':
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError

        self.fc_mid = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_channels // reduction, out_channels * 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_2d, feat_3d):
        bs = feat_2d.shape[0]

        feat_2d = self.align1(feat_2d)
        feat_3d = self.align2(feat_3d)

        weight = self.avg_pool(feat_2d + feat_3d).reshape([bs, -1])  # [bs, C]
        weight = self.fc_mid(weight)  # [bs, C / r]
        weight = self.fc_out(weight).reshape([bs, -1, 2])  # [bs, C, 2]
        weight = softmax(weight, dim=-1)
        w1, w2 = weight[..., 0], weight[..., 1]  # [bs, C]

        if len(feat_2d.shape) == 4:
            w1 = w1.reshape([bs, -1, 1, 1])
            w2 = w2.reshape([bs, -1, 1, 1])
        else:
            w1 = w1.reshape([bs, -1, 1])
            w2 = w2.reshape([bs, -1, 1])

        return feat_2d * w1 + feat_3d * w2
