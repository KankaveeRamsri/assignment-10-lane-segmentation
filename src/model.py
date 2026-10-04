"""TinyLaneSeg: a small encoder-decoder for lane segmentation, designed from scratch.

All weights are randomly initialised (Kaiming) -- no pretrained backbone is used.

Design summary (see README.md for the full rationale):
  * Stem: two plain 3x3 convs (stride 2) -> 1/2 resolution, cheap early features.
  * Encoder: stacks of depthwise-separable residual blocks, stride 2 per stage
    -> 1/4, 1/8, 1/16 resolution. Depthwise-separable convs cost ~8-9x fewer
    parameters/FLOPs than plain 3x3 convs at the same width.
  * Context module at 1/16: parallel dilated depthwise branches (d = 1, 2, 4, 8)
    plus a global-pooling branch. Lanes are long, thin structures that span the
    whole frame, so the model needs a large receptive field without more downsampling.
  * Decoder: bilinear upsampling + skip connections from 1/8 and 1/4 to recover
    sharp lane boundaries, then a light head and a final x4 upsample.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(cin, cout, k=3, s=1, d=1, groups=1):
    pad = d * (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, pad, dilation=d, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DSBlock(nn.Module):
    """Depthwise-separable residual block: DW3x3 -> PW1x1 -> DW3x3 -> PW1x1 (+ shortcut)."""

    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn_act(cin, cin, 3, stride, dilation, groups=cin),
            conv_bn_act(cin, cout, 1),
            nn.Conv2d(cout, cout, 3, 1, dilation, dilation=dilation, groups=cout, bias=False),
            nn.BatchNorm2d(cout),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        if stride == 1 and cin == cout:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        return F.relu(self.body(x) + self.short(x), inplace=True)


class ContextModule(nn.Module):
    """Multi-dilation context aggregation (a lightweight ASPP-style module)."""

    def __init__(self, c, branch_c, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(conv_bn_act(c, c, 3, 1, d, groups=c), conv_bn_act(c, branch_c, 1))
            for d in dilations
        )
        # no BatchNorm on the 1x1 pooled vector (it breaks with batch size 1)
        self.gpool = nn.Sequential(nn.Conv2d(c, branch_c, 1), nn.ReLU(inplace=True))
        self.fuse = conv_bn_act(branch_c * (len(dilations) + 1), c, 1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        g = self.gpool(F.adaptive_avg_pool2d(x, 1))
        outs.append(g.expand(-1, -1, x.shape[2], x.shape[3]))
        return self.fuse(torch.cat(outs, 1)) + x


class TinyLaneSeg(nn.Module):
    def __init__(self, num_classes=3, widths=(16, 32, 64, 96, 128)):
        super().__init__()
        c0, c1, c2, c3, c4 = widths
        self.stem = nn.Sequential(conv_bn_act(3, c0, 3, 2), conv_bn_act(c0, c1, 3, 1))  # 1/2
        self.stage1 = nn.Sequential(DSBlock(c1, c2, 2), DSBlock(c2, c2))                # 1/4
        self.stage2 = nn.Sequential(DSBlock(c2, c3, 2), DSBlock(c3, c3))                # 1/8
        self.stage3 = nn.Sequential(DSBlock(c3, c4, 2), DSBlock(c4, c4, dilation=2))    # 1/16
        self.context = ContextModule(c4, 32)

        self.skip8 = conv_bn_act(c3, 64, 1)
        self.dec8 = DSBlock(c4 + 64, 64)
        self.skip4 = conv_bn_act(c2, 32, 1)
        self.dec4 = DSBlock(64 + 32, 48)
        self.head = nn.Sequential(conv_bn_act(48, 32, 3), nn.Conv2d(32, num_classes, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.stem(x)
        f4 = self.stage1(x)
        f8 = self.stage2(f4)
        f16 = self.context(self.stage3(f8))

        y = F.interpolate(f16, size=f8.shape[2:], mode="bilinear", align_corners=False)
        y = self.dec8(torch.cat([y, self.skip8(f8)], 1))
        y = F.interpolate(y, size=f4.shape[2:], mode="bilinear", align_corners=False)
        y = self.dec4(torch.cat([y, self.skip4(f4)], 1))
        y = self.head(y)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TinyLaneSeg()
    out = m(torch.randn(1, 3, 256, 448))
    print(out.shape, f"{count_params(m):,} params")
