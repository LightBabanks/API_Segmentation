import torch
import torch.nn as nn
import torch.nn.functional as F

from models.architecture_unet import DoubleConv


class HaarDWT2D(nn.Module):
    def forward(self, x: torch.Tensor):
        assert x.shape[-2] % 2 == 0 and x.shape[-1] % 2 == 0, "H et W doivent être pairs"
        z = F.pixel_unshuffle(x, 2)
        x00, x01, x10, x11 = torch.chunk(z, 4, dim=1)

        ll = (x00 + x01 + x10 + x11) / 2.0
        lh = (x00 - x01 + x10 - x11) / 2.0
        hl = (x00 + x01 - x10 - x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor):
        x00 = (ll + lh + hl + hh) / 2.0
        x01 = (ll - lh + hl - hh) / 2.0
        x10 = (ll + lh - hl - hh) / 2.0
        x11 = (ll - lh - hl + hh) / 2.0
        z = torch.cat([x00, x01, x10, x11], dim=1)
        return F.pixel_shuffle(z, 2)


class SubbandProcessor(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DWTEnhancedSkip(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.proc_ll = SubbandProcessor(channels)
        self.proc_lh = SubbandProcessor(channels)
        self.proc_hl = SubbandProcessor(channels)
        self.proc_hh = SubbandProcessor(channels)

        self.w_ll = nn.Parameter(torch.tensor(1.0))
        self.w_lh = nn.Parameter(torch.tensor(1.0))
        self.w_hl = nn.Parameter(torch.tensor(1.0))
        self.w_hh = nn.Parameter(torch.tensor(1.0))

        self.refine = DoubleConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(x)

        ll = self.w_ll * self.proc_ll(ll)
        lh = self.w_lh * self.proc_lh(lh)
        hl = self.w_hl * self.proc_hl(hl)
        hh = self.w_hh * self.proc_hh(hh)

        x_rec = self.idwt(ll, lh, hl, hh)
        return self.refine(x_rec)


class FreqDWTUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.skip4 = DWTEnhancedSkip(base * 8)
        self.skip3 = DWTEnhancedSkip(base * 4)
        self.skip2 = DWTEnhancedSkip(base * 2)
        self.skip1 = DWTEnhancedSkip(base)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        s4 = self.skip4(e4)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        s3 = self.skip3(e3)
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        s2 = self.skip2(e2)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        s1 = self.skip1(e1)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.head(d1)
