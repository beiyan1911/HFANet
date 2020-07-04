import torch
import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, kernel_size, stride, padding, bias=False):
        super(BaseBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size, stride, padding, bias=bias)
        )

    def forward(self, x):
        return self.model(x)


class SFBlock(nn.Module):
    def __init__(self, n_feats):
        super(SFBlock, self).__init__()

        kernel_size = 3
        stride = 1
        pad = 1

        self.body1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class HDFBlock(nn.Module):
    def __init__(self, n_feats=16, step=6):
        super(HDFBlock, self).__init__()

        self.gap_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats * 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv2d(n_feats * 3, step, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.gmp_attention = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(n_feats, n_feats * 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.Conv2d(n_feats * 3, step, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

        self.step = step
        self._ops = nn.ModuleList()
        self.out = nn.Sequential(nn.Conv2d(n_feats * step, n_feats, 1, padding=0, bias=False), nn.ReLU(inplace=True))
        for _ in range(self.step):
            self._ops.append(BaseBlock(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        gap = self.gap_attention(x)
        gmp = self.gap_attention(x)
        scale = self.softmax(gap + gmp)
        s0 = x
        res = []
        for i in range(self.step):
            weight = scale[:, i:i + 1]
            s0 = self._ops[i](s0)
            res.append(s0 * weight)
        out = self.out(torch.cat(res, dim=1))
        return x + out


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, stride, padding, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class DABlock(nn.Module):
    def __init__(self, n_feats, step=5):
        super(DABlock, self).__init__()
        self.sf = SFBlock(n_feats)
        self.hdf = HDFBlock(n_feats, step)

    def forward(self, x):
        x_out = self.sf(x)
        x_out = self.hdf(x_out)
        return x_out


class HFANetWork(nn.Module):

    def __init__(self, in_dims=3, n_feats=24, step=5, layer_num=15):
        super(HFANetWork, self).__init__()

        kernel_size = 3

        self.FEB = nn.Sequential(nn.Conv2d(in_dims, n_feats, kernel_size, stride=1, padding=1, bias=False),
                                 ResBlock(n_feats, kernel_size, stride=1, padding=1, bias=False),
                                 ResBlock(n_feats, kernel_size, stride=1, padding=1, bias=False),
                                 ResBlock(n_feats, kernel_size, stride=1, padding=1, bias=False),
                                 ResBlock(n_feats, kernel_size, stride=1, padding=1, bias=False), )
        self.layer_num = layer_num

        self.layers = nn.ModuleList()

        for i in range(self.layer_num):
            layer = DABlock(n_feats, step)

            self.layers.append(layer)

        self.tail = nn.Conv2d(n_feats, 3, kernel_size, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.FEB(x)
        for i in range(self.layer_num):
            out = self.layers[i](out)
        out = self.tail(out)
        return out
