import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block

class Basic_cnn_block(nn.Module):
    """
    Building the basic modules of the network backbone.
    You can choose our method or unet.

    Args:
        blk_cha (str, optional): Execute the emcam or unet module.
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels
    """
    def __init__(self, blk_cha, in_chans, out_chans):
        super().__init__()
        self.blk_cha = blk_cha
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding="same"),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
        )
        self.split = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same"),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
        )
        self.transform1 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same"),
            nn.GELU(),
        )
        self.transform2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same", dilation=2),
            nn.GELU()
        )
        self.transform3 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same", dilation=3),
            nn.GELU()
        )
        self.extension1 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same"),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )
        self.extension2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, 3, 1, padding="same"),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )

    def forward(self, x):
        if self.blk_cha == 'unet':
            x_out = self.double_conv(x)
        elif self.blk_cha == "emca":
            x = self.expand(x)
            residual = self.split(x)
            x1 = self.transform1(residual)
            x1 = F.dropout(x1, 0.1)
            x2 = self.transform2(residual)
            x2 = F.dropout(x2, 0.1)
            x3 = self.transform3(residual)
            x3 = F.dropout(x3, 0.1)
            add_x1_x2 = torch.add(x1, x2)
            add = torch.add(add_x1_x2, x3)
            x_out = self.extension1(add)
            x_out = F.dropout(x_out, 0.1)
            x_out = residual + x_out
            x_out = self.extension2(x_out)

        return x_out

class Encoder(nn.Module):
    """
    Build encoder.

    Agrs:
        blk_cha (str, optional): Execute the emcam or unet module.
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels
    """
    def __init__(self, blk_cha, in_chans, out_chans):
        super().__init__()
        self.blk_cha = blk_cha
        # in_chans = [3, 32, 64, 128, 256]
        # out_chans = [32, 64, 128, 256, 512]
        self.blk_cha = blk_cha
        self.e_stage1 = Basic_cnn_block(self.blk_cha, in_chans[0], out_chans[0])
        self.e_stage2 = Basic_cnn_block(self.blk_cha, in_chans[1], out_chans[1])
        self.e_stage3 = Basic_cnn_block(self.blk_cha, in_chans[2], out_chans[2])
        self.e_stage4 = Basic_cnn_block(self.blk_cha, in_chans[3], out_chans[3])
        self.e_stage5 = Basic_cnn_block(self.blk_cha, in_chans[4], out_chans[4])
    def forward(self, x):
        skip = []
        # stage1
        x1 = self.e_stage1(x)
        x2 = F.max_pool2d(x1, (2, 2))
        skip.append(x2)

        # stage2
        x2 = self.e_stage2(x2)
        x3 = F.max_pool2d(x2, (2, 2))
        skip.append(x3)

        # stage3
        x3 = self.e_stage3(x3)
        x4 = F.max_pool2d(x3, (2, 2))
        skip.append(x4)

        # stage4
        x4 = self.e_stage4(x4)
        x5 = F.max_pool2d(x4, (2, 2))
        skip.append(x5)

        # stage5
        x5 = self.e_stage5(x5)
        x_out = F.max_pool2d(x5, (2, 2))

        return x_out, skip

class Skip_Bottleneck_Layer(nn.Module):
    """
    Bulid modeling global representation module.
    This module is used in skip connections and bottleneck layers.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        input_n (int): Position code length.
        num_heads (int): Number of attention heads.
        depth (int): Number of blocks.
    """
    def __init__(self, dim, input_resolution, input_n, num_heads, depth):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.input_n = input_n
        self.num_heads = num_heads
        self.depth = depth
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]
        drop_path = dpr[sum(self.depth[:0]):sum(self.depth[:1])]
        self.pos_embed_bottm = nn.Parameter(torch.randn(1, self.input_n[0], self.dim[0]) * .02)
        self.bottm = nn.ModuleList([
            Block(dim=self.dim[0], num_heads=self.num_heads[0], mlp_ratio=4., qkv_bias=True,
                  init_values=None, attn_drop=0., drop_path=drop_path[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth[0])])

        drop_path4 = dpr[sum(self.depth[:1]):sum(self.depth[:2])]
        self.pos_embed_4 = nn.Parameter(torch.randn(1, self.input_n[1], self.dim[1]) * .02)
        self.skip4 = nn.ModuleList([
            Block(dim=self.dim[1], num_heads=self.num_heads[1], mlp_ratio=4., qkv_bias=True,
                  init_values=None, attn_drop=0., drop_path=drop_path4[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth[1])])

        drop_path3 = dpr[sum(self.depth[:2]):sum(self.depth[:3])]
        self.pos_embed_3 = nn.Parameter(torch.randn(1, self.input_n[2], self.dim[2]) * .02)
        self.skip3 = nn.ModuleList([
            Block(dim=self.dim[2], num_heads=self.num_heads[2], mlp_ratio=4., qkv_bias=True,
                  init_values=None, attn_drop=0., drop_path=drop_path3[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth[2])])

        drop_path2 = dpr[sum(self.depth[:3]):sum(self.depth[:4])]
        self.pos_embed_2 = nn.Parameter(torch.randn(1, self.input_n[3], self.dim[3]) * .02)
        self.skip2 = nn.ModuleList([
            Block(dim=self.dim[3], num_heads=self.num_heads[3], mlp_ratio=4., qkv_bias=True,
                  init_values=None, attn_drop=0., drop_path=drop_path2[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth[3])])

        # drop_path1 = dpr[sum(self.depth[:4]):sum(self.depth[:5])]
        # self.pos_embed_1 = nn.Parameter(torch.randn(1, self.input_n[4], self.dim[4]) * .02)
        # self.skip1 = nn.ModuleList([
        #      Block(dim=self.dim[4], num_heads=self.num_heads[4], mlp_ratio=4., qkv_bias=True,
        #            init_values=None, drop=0.,attn_drop=0., drop_path=drop_path1[i],
        #            norm_layer=norm_layer, act_layer=act_layer)
        #      for i in range(self.depth[4])])

    def forward(self, x, skip):
        skip_out = []

        # skip_bottm
        x_bottm = x.flatten(2).transpose(1,2)
        x_bottm = x_bottm + self.pos_embed_bottm
        for blk in self.bottm:
            x_bottm = blk(x_bottm)
        H, W = self.input_resolution[0]
        B, L, C = x_bottm.shape
        assert L == H * W, "input feature has wrong size"
        x_bottm = x_bottm.view(B, H, W, C).permute(0, 3, 1, 2)

        # skip4
        skip_4 = skip[3].flatten(2).transpose(1,2)
        skip_4 = skip_4 + self.pos_embed_4
        for blk4 in self.skip4:
            skip_4 = blk4(skip_4)
        H, W = self.input_resolution[1]
        B, L, C = skip_4.shape
        assert L == H * W, "input feature has wrong size"
        skip_4 = skip_4.view(B, H, W, C).permute(0, 3, 1, 2)

        # skip3
        skip_3 = skip[2].flatten(2).transpose(1, 2)
        skip_3 = skip_3 + self.pos_embed_3
        for blk3 in self.skip3:
            skip_3 = blk3(skip_3)
        H, W = self.input_resolution[2]
        B, L, C = skip_3.shape
        assert L == H * W, "input feature has wrong size"
        skip_3 = skip_3.view(B, H, W, C).permute(0, 3, 1, 2)

        # skip2
        skip_2 = skip[1].flatten(2).transpose(1, 2)
        skip_2 = skip_2 + self.pos_embed_2
        for blk2 in self.skip2:
            skip_2 = blk2(skip_2)
        H, W = self.input_resolution[3]
        B, L, C = skip_2.shape
        assert L == H * W, "input feature has wrong size"
        skip_2 = skip_2.view(B, H, W, C).permute(0, 3, 1, 2)

        # skip1
        skip_1 = skip[0]
        #     .flatten(2).transpose(1, 2)
        # skip_1 += self.pos_embed_1
        # for blk1 in self.skip1:
        #     skip_1 = blk1(skip_1)
        # H, W = self.input_resolution[4]
        # B, L, C = skip_1.shape
        # assert L == H * W, "input feature has wrong size"
        # skip_1 = skip_1.view(B, H, W, C).permute(0, 3, 1, 2)

        skip_out.append(skip_1)
        skip_out.append(skip_2)
        skip_out.append(skip_3)
        skip_out.append(skip_4)

        return x_bottm, skip_out


class Decoder(nn.Module):
    """
    Build encoder.

    Agrs:
        blk_cha (str, optional): Execute the emcam or unet module.
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        bilinear (bool): Whether to up-sample.
    """
    def __init__(self, in_chans, out_chans, blk_cha, bilinear=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.blk_cha = blk_cha
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.up = nn.ConvTranspose2d(in_channels=out_chans[4], out_channels=out_chans[4], kernel_size=2, stride=2)
        self.e_stage4 = Basic_cnn_block(self.blk_cha, out_chans[4] + in_chans[4], in_chans[3])
        self.e_stage3 = Basic_cnn_block(self.blk_cha, out_chans[3], in_chans[2])
        self.e_stage2 = Basic_cnn_block(self.blk_cha, out_chans[2], in_chans[1])
        self.e_stage1 = Basic_cnn_block(self.blk_cha, out_chans[1], in_chans[1])
    def forward(self, x_bottm, skip_out):
        # bottm
        x_4 = self.upsample(x_bottm)
        x_4 = self.e_stage4(torch.cat((x_4, skip_out[3]), dim=1))

        # decoder stage4
        x_3 = self.upsample(x_4)
        x_3 = self.e_stage3(torch.cat((x_3, skip_out[2]), dim=1))

        # decoder stage3
        x_2 = self.upsample(x_3)
        x_2 = self.e_stage2(torch.cat((x_2, skip_out[1]), dim=1))

        # decoder stage2
        x_1 = self.upsample(x_2)
        x_1 = self.e_stage1(torch.cat((x_1, skip_out[0]), dim=1))
        x_decoder = self.upsample(x_1)
        return x_decoder

class Seg_Head(nn.Module):
    """
        Build segmentation head.

        Agrs:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
        """
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv = nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, stride=1, padding="same")
        self.batch_norm = nn.BatchNorm2d(self.in_chans)
        self.seg = nn.Conv2d(self.in_chans, self.out_chans, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv(x)))
        x_out = self.seg(x)

        return x_out

class EMCAMNet(nn.Module):
    """
        Build EMCAM-Net.

        Agrs:
            num_classes (int): Class number of segmentation.
        """
    def __init__(self, num_classes=9):
        super().__init__()
        # encoder-decoder
        in_chans = [3, 32, 64, 128, 256]
        out_chans = [32, 64, 128, 256, 512]
        blk_cha = "emca"
        bilinear = True
        # MGR
        dim = [512, 256, 128,64]
        # ACDC/Synapse
        input_resolution = [(7, 7), (14, 14), (28, 28), (56, 56)]
        input_n = [49, 196, 784, 3136]
        # DRIVE
        # input_resolution = [(18, 18), (36, 36), (72, 72), (144, 144)]
        # input_n = [324, 1296, 5184, 20736]
        num_head = [16, 8, 8, 4]
        depth = [4, 1, 1, 1]
        self.encoder = Encoder(blk_cha, in_chans, out_chans)
        self.skip_bottleneck = Skip_Bottleneck_Layer(dim, input_resolution, input_n, num_head, depth)
        self.decoder = Decoder(in_chans, out_chans, blk_cha, bilinear)
        self.seg = Seg_Head(in_chans=in_chans[1], out_chans=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, skip = self.encoder(x)
        x_bottm, skip_out = self.skip_bottleneck(x, skip)
        x_decoder = self.decoder(x_bottm, skip_out)
        out = self.seg(x_decoder)
        return out