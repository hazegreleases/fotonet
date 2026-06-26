import torch
import torch.nn as nn


def _valid_num_heads(dim, requested):
    """Pick an attention head count that divides dim for arbitrary tiny widths."""
    requested = max(int(requested), 1)
    for heads in range(min(requested, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class Conv(nn.Module):
    """Standard convolution-BN-SiLU block."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p or k // 2, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C3k(nn.Module):
    """C3k block with 3x3 convs."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, k, 1, g=g) for _ in range(n)))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        return x + y if self.add else y

class C3k2(nn.Module):
    """C3k2 block (YOLOv11 style)."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(C3k(c_, c_, shortcut=shortcut, g=g, k=3 if c3k else 1) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    """Spatial attention module."""
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = _valid_num_heads(dim, num_heads)
        self.head_dim = dim // self.num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * self.num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.key_dim * self.num_heads, self.key_dim * self.num_heads, C], 1)
        q = q.view(B, self.num_heads, self.key_dim, N).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.key_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(-1, -2).reshape(B, C, H, W)
        x = self.proj(x) + self.pe(x)
        return x

class PSABlock(nn.Module):
    """PSA block (YOLOv11 style)."""
    def __init__(self, c):
        super().__init__()
        heads = max(c // 32, 1)
        self.attn = Attention(c, num_heads=heads)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1, 1),
            Conv(c * 2, c, 1, 1)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class C2PSA(nn.Module):
    """C2PSA module (YOLOv11 style)."""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(PSABlock(c_) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class SPPF(nn.Module):
    """SPPF layer."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class SCDown(nn.Module):
    """SCDown (YOLOv10 style)."""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = nn.Conv2d(c2, c2, k, s, k // 2, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cv2(self.cv1(x))))

class CIB(nn.Module):
    """CIB (YOLOv10 style)."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c2, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ChannelGate(nn.Module):
    """Cheap P5 channel reweighting: adds capacity with almost no spatial FLOPs."""
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(self.pool(x))


class C2fCIB(nn.Module):
    """C2fCIB (YOLOv10 style)."""
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Backbone(nn.Module):
    """SOTA Backbone."""
    def __init__(
        self,
        w=0.25,
        d=0.33,
        use_p2=False,
        p3_extra_blocks=0,
        p4_extra_blocks=0,
        p5_extra_blocks=0,
        p5_gate_blocks=0,
    ):
        super().__init__()
        self.use_p2 = bool(use_p2)
        self.p3_extra_blocks = max(int(p3_extra_blocks), 0)
        self.p4_extra_blocks = max(int(p4_extra_blocks), 0)
        self.p5_extra_blocks = max(int(p5_extra_blocks), 0)
        self.p5_gate_blocks = max(int(p5_gate_blocks), 0)
        ch = [int(x * w) for x in [64, 128, 256, 512, 1024]]
        self.stem = Conv(3, ch[0], k=3, s=2)
        self.stage1 = nn.Sequential(SCDown(ch[0], ch[1]), C3k2(ch[1], ch[1], n=max(round(3*d),1)))
        self.stage2 = nn.Sequential(
            SCDown(ch[1], ch[2]),
            C3k2(ch[2], ch[2], n=max(round(6*d),1)),
            *(CIB(ch[2], ch[2], shortcut=True, e=0.5) for _ in range(self.p3_extra_blocks)),
        )
        self.stage3 = nn.Sequential(
            SCDown(ch[2], ch[3]),
            C3k2(ch[3], ch[3], n=max(round(6*d),1), c3k=True),
            *(CIB(ch[3], ch[3], shortcut=True, e=0.5) for _ in range(self.p4_extra_blocks)),
        )
        p5_tail = [CIB(ch[4], ch[4], shortcut=True, e=0.5) for _ in range(self.p5_extra_blocks)]
        p5_tail.extend(ChannelGate(ch[4]) for _ in range(self.p5_gate_blocks))
        self.stage4 = nn.Sequential(
            SCDown(ch[3], ch[4]),
            C3k2(ch[4], ch[4], n=max(round(3*d),1), c3k=True),
            SPPF(ch[4], ch[4]),
            C2PSA(ch[4], ch[4]),
            *p5_tail,
        )
        self.out_channels = (ch[1], ch[2], ch[3], ch[4]) if self.use_p2 else (ch[2], ch[3], ch[4])

    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        if self.use_p2:
            return p2, p3, p4, p5
        return p3, p4, p5
