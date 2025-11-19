import torch
import torch.nn as nn
import torch.nn.functional as F
from .wavelet import *


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return self.weight * x


class _WTCore(nn.Module):
    def __init__(self, hidden_channels, kernel_size=5, bias=True, wt_levels=1, wt_type='db1', stride=1):
        super().__init__()
        self.hidden = hidden_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_flt, iwt_flt = wavelet.create_2d_wavelet_filter(
            wt_type, hidden_channels, hidden_channels, torch.float
        )
        self.wt_filter = nn.Parameter(wt_flt, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_flt, requires_grad=False)

        self.base_conv = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size,
            padding='same', stride=1, dilation=1,
            groups=hidden_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, hidden_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                hidden_channels * 4, hidden_channels * 4, kernel_size,
                padding='same', stride=1, dilation=1,
                groups=hidden_channels * 4, bias=False
            ) for _ in range(wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, hidden_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(wt_levels)
        ])

        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride) if stride > 1 else None

    def _mask_subbands(self, ll, h3, mode: str):
        if mode == 'low':
            h3 = torch.zeros_like(h3)
        elif mode == 'high':
            ll = torch.zeros_like(ll)
        else:
            raise ValueError("mode must be 'low' or 'high'")
        return ll, h3

    def forward_band(self, x, mode: str):
        x_ll_stack, x_h_stack, shapes = [], [], []
        curr_ll = x

        for i in range(self.wt_levels):
            shp = curr_ll.shape
            shapes.append(shp)
            if (shp[2] % 2) or (shp[3] % 2):
                curr_ll = F.pad(curr_ll, (0, shp[3] % 2, 0, shp[2] % 2))

            wt = wavelet.wavelet_2d_transform(curr_ll, self.wt_filter)   # (B,C,4,H/2,W/2)
            curr_ll = wt[:, :, 0, :, :]

            b, c, four, h, w = wt.shape
            tag = wt.reshape(b, c * 4, h, w)
            tag = self.wavelet_scale[i](self.wavelet_convs[i](tag))
            tag = tag.reshape(b, c, 4, h, w)

            x_ll_stack.append(tag[:, :, 0, :, :])
            x_h_stack.append(tag[:, :, 1:4, :, :])

        next_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            ll = x_ll_stack[i]
            h3 = x_h_stack[i]
            ll, h3 = self._mask_subbands(ll, h3, mode=mode)

            wt_cat = torch.cat([ll.unsqueeze(2), h3], dim=2)
            rec = wavelet.inverse_2d_wavelet_transform(wt_cat, self.iwt_filter)
            shp = shapes[i]
            rec = rec[:, :, :shp[2], :shp[3]]
            next_ll = rec + (next_ll if isinstance(next_ll, torch.Tensor) else 0)

        band = next_ll
        band = self.base_scale(self.base_conv(band)) + band
        if self.do_stride is not None:
            band = self.do_stride(band)
        return band


class WTLowPassConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True,
                 wt_levels=1, wt_type='db1', hidden_channels=None):
        super().__init__()
        hidden = hidden_channels if hidden_channels is not None else max(in_channels, out_channels)
        self.pre = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.core = _WTCore(hidden, kernel_size=kernel_size, bias=bias,
                            wt_levels=wt_levels, wt_type=wt_type, stride=stride)
        self.post = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True)
    def forward(self, x):
        h = self.pre(x)
        h = self.core.forward_band(h, mode='low')
        return self.post(h)


class WTHighPassConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True,
                 wt_levels=1, wt_type='db1', hidden_channels=None):
        super().__init__()
        hidden = hidden_channels if hidden_channels is not None else max(in_channels, out_channels)
        self.pre = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.core = _WTCore(hidden, kernel_size=kernel_size, bias=bias,
                            wt_levels=wt_levels, wt_type=wt_type, stride=stride)
        self.post = nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True)
    def forward(self, x):
        h = self.pre(x)
        h = self.core.forward_band(h, mode='high')
        return self.post(h)
