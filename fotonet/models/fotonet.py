"""
FOTO-NET top-level model assembly.
Wires backbone to neck to head, passing channel dimensions automatically.
"""
import torch
import torch.nn as nn
from fotonet.models.backbone import Backbone
from fotonet.models.neck import Neck
from fotonet.models.head import FOTONETHead


def _init_weights(m):
    """Better weight initialization for faster convergence: He/Kaiming for convs, normal for biases."""
    if isinstance(m, nn.Conv2d):
        # SiLU ≈ leaky_relu in terms of gain; 'relu' gives wrong scale for SiLU networks
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FOTONETModel(nn.Module):
    def __init__(
        self,
        nc=80,
        w=1.0,
        d=1.0,
        imgsz=640,
        use_p2=True,
        reg_max=1,
        p3_extra_blocks=0,
        p4_extra_blocks=0,
        p5_extra_blocks=0,
        p5_gate_blocks=0,
        arch_version=1,
        neck_fusion="concat",
        p2_context_blocks=0,
        p3_context_blocks=0,
        quality_head=False,
    ):
        super().__init__()
        self.w = w
        self.d = d
        self.use_p2 = bool(use_p2)
        self.reg_max = int(reg_max)
        self.p3_extra_blocks = max(int(p3_extra_blocks), 0)
        self.p4_extra_blocks = max(int(p4_extra_blocks), 0)
        self.p5_extra_blocks = max(int(p5_extra_blocks), 0)
        self.p5_gate_blocks = max(int(p5_gate_blocks), 0)
        self.arch_version = int(arch_version)
        self.neck_fusion = str(neck_fusion)
        self.p2_context_blocks = max(int(p2_context_blocks), 0)
        self.p3_context_blocks = max(int(p3_context_blocks), 0)
        self.quality_head = bool(quality_head)
        self.width_multiple = w
        self.depth_multiple = d
        self.backbone = Backbone(
            w=w,
            d=d,
            use_p2=self.use_p2,
            p3_extra_blocks=self.p3_extra_blocks,
            p4_extra_blocks=self.p4_extra_blocks,
            p5_extra_blocks=self.p5_extra_blocks,
            p5_gate_blocks=self.p5_gate_blocks,
        )
        self.neck     = Neck(
            in_channels=self.backbone.out_channels,
            d=d,
            use_p2=self.use_p2,
            fusion=self.neck_fusion,
            p2_context_blocks=self.p2_context_blocks,
            p3_context_blocks=self.p3_context_blocks,
        )
        strides       = [4, 8, 16, 32] if self.use_p2 else [8, 16, 32]
        self.head     = FOTONETHead(
            nc=nc,
            in_channels=self.neck.out_channels,
            imgsz=imgsz,
            strides=strides,
            reg_max=self.reg_max,
            quality_head=self.quality_head,
        )
        
        # Apply general weight initialization first
        self.apply(_init_weights)
        
        # CRITICAL: Re-apply cls head bias AFTER general init
        # (_init_weights resets all biases to 0, but cls heads need -4.595
        #  so initial foreground confidence is ~1% instead of 50%)
        self.head._init_cls_bias()

    def forward(self, x, return_all=False, use_o2m=None, return_preview=False):
        # Derive actual image size from the input tensor so box decoding is always correct,
        # even when multi-scale training (ERA half-res) is in use.
        # We use a tensor here for CUDA Graphs compatibility.
        actual_imgsz = torch.as_tensor(x.shape[-1], device=x.device, dtype=torch.float32)
        feats        = self.backbone(x)
        p_feats      = self.neck(feats)
        out = self.head(p_feats, imgsz=actual_imgsz, return_all=return_all, use_o2m=use_o2m)
        if return_preview and isinstance(out, dict) and p_feats:
            out["preview_features"] = p_feats[0].detach()
        return out

    def strip_o2m_for_inference(self):
        self.head.strip_o2m_for_inference()
        return self
