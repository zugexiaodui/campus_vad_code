import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn import functional as tf
from torchvision.transforms import functional as tvf
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, load_state_dict_from_url, model_urls
from typing import Type, Any, Callable, Union, List, Optional, Tuple
from functools import partial
from einops import rearrange

from torchvision.models.resnet import BasicBlock, Bottleneck
from module_cvae import CVAE

feat_dim_dict = {'basic': (64, 64, 128, 256, 512),
                 'bottleneck': (64, 256, 512, 1024, 2048)}
feat_size_list = (128, 64, 32, 16, 8)


class BasicBlockUpsample(BasicBlock):
    '''
    Here `downsample` is actually an upsample
    '''

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, relu_type=nn.LeakyReLU, conv1: nn.Module = None) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        if conv1:
            if issubclass(conv1, nn.ConvTranspose2d):
                self.conv1 = conv1(inplanes, planes, kernel_size=stride, stride=stride, bias=False)
            else:
                raise TypeError(type(conv1))
        if downsample:
            if issubclass(conv1, nn.ConvTranspose2d):
                self.downsample = downsample(inplanes, planes, kernel_size=stride, stride=stride, bias=False)
            else:
                raise TypeError(type(conv1))

        self.relu = relu_type()


class BottleneckUpsample(Bottleneck):
    '''
    Here `downsample` is actually an upsample
    '''
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, relu_type=nn.LeakyReLU, conv1: nn.Module = None) -> None:
        super().__init__(inplanes, planes, 1, downsample, groups, base_width, dilation, norm_layer)
        if conv1:
            if issubclass(conv1, nn.ConvTranspose2d):
                self.conv1 = conv1(inplanes, planes, kernel_size=stride, stride=stride, bias=False)
            else:
                raise TypeError(type(conv1))
        if downsample:
            if issubclass(conv1, nn.ConvTranspose2d):
                self.downsample = downsample(inplanes, planes, kernel_size=stride, stride=stride, bias=False)
            else:
                raise TypeError(type(conv1))

        self.relu = relu_type()


class FrameDecoder(nn.Module):
    def __init__(self, encoder_block_type: str, decoder_block_type: str, num_frm: int = 1):
        super().__init__()

        encoder_block_type = encoder_block_type.lower()
        if encoder_block_type == 'basic':
            _ch_div = 1
        elif encoder_block_type == 'bottleneck':
            _ch_div = 4
        else:
            raise NotImplementedError(f"{encoder_block_type}")
        feat_dim = list(feat_dim_dict[encoder_block_type])
        feat_dim.reverse()

        decoder_block_type = decoder_block_type.lower()
        if decoder_block_type == 'basic':
            block_type = BasicBlockUpsample
        elif decoder_block_type == 'bottleneck':
            block_type = BottleneckUpsample
        else:
            raise NotImplementedError(f"{decoder_block_type}")

        upconv = nn.ConvTranspose2d
        relu_type = nn.LeakyReLU
        norm_layer = nn.BatchNorm2d

        up_block = partial(block_type, stride=2, conv1=upconv, downsample=upconv, relu_type=relu_type, norm_layer=norm_layer)

        # [1, 512+512, 32, 32]
        self.uconv3 = nn.Sequential(up_block(feat_dim[2], c := feat_dim[2] // 2), BasicBlock(c, c))  # [1, 256, 64, 64]

        # [1, 256+256, 64, 64]
        self.uconv4 = nn.Sequential(up_block(feat_dim[3] * 2, c := feat_dim[3] // _ch_div), BasicBlock(c, c))  # [1, 64, 128, 128]

        # [1, 64+64, 128, 128]
        self.uconv5 = nn.Sequential(up_block(feat_dim[4], c := feat_dim[4] // 2), BasicBlock(c, c))  # [1, 32, 256, 256]

        # [1, 32, 256, 256]
        _rec_ch = feat_dim[4] // 2

        self.num_frm = num_frm
        self.rec_conv = nn.Sequential(
            nn.Conv2d(_rec_ch, _rec_ch, 3, 1, 1, bias=False),
            norm_layer(_rec_ch),
            relu_type(),
            nn.Conv2d(_rec_ch, 3 * self.num_frm, 3, 1, 1)
        )  # [1, 3, 256, 256]

        self.init_params()

    def init_params(self, init_type='kaiming'):
        init_gain = 1.0

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, feat_list: List[Tensor]):
        # feat_list[i]: [b, c, h, w]
        y2, y3 = feat_list
        z2 = self.uconv3(y3)  # (512+512,32)->(256,64)
        z1 = self.uconv4(torch.cat([z2, y2], 1))  # (256+256,64)->(64,128)
        z0 = self.uconv5(z1)  # (64+64,128)->(32,256)
        out = self.rec_conv(z0)  # (32,256)->(3,256)
        out = rearrange(out, 'b (c t) h w -> b c t h w', c=3, t=self.num_frm)
        return out


class FrameEncoder(ResNet):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], img_dim: int = 3, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__(block, layers, 0, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

        self.conv1 = nn.Conv2d(img_dim, self.conv1.out_channels, kernel_size=self.conv1.kernel_size, stride=self.conv1.stride, padding=self.conv1.padding, bias=self.conv1.bias)

        del self.layer3
        del self.layer4
        del self.avgpool
        del self.fc

        self.init_params()

    def init_params(self, init_type='kaiming'):
        init_gain = 1.0

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        feat_list = []
        x = self.conv1(x)  # (64,128)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # (256,64)
        feat_list.append(x)
        x = self.layer2(x)  # (512,32)
        feat_list.append(x)

        return feat_list


class BackgroundEncoder(nn.Module):
    def __init__(self, num_classes: int, img_dim=3):
        super().__init__()

        backbone = nn.Sequential(
            # 480
            nn.Conv2d(3, 32, 7, 4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 7, 4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, 2),  # 13
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.num_classes = num_classes
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inp_bgd: Tensor, only_feat: bool = False):
        # inp_bgd: [b, c, h, w]
        inp_bgd = tvf.resize(inp_bgd, (480, 480))
        feat: Tensor = self.backbone(inp_bgd)  # (b, 128, 13, 13)
        if only_feat:
            return feat
        feat: Tensor = self.avg_pool(feat)
        feat.squeeze_(2).squeeze_(2)  # (b, 128)
        out = self.classifier(feat)
        return out


class SceneFrameAE(nn.Module):
    def __init__(self, img_dim: int = 3, inp_frm: int = 8, tgt_frm: int = 1, bgd_encoder: BackgroundEncoder = None, encoder_block_type: str = 'basic', decoder_block_type='basic', pretrained_encoder: bool = False, writer=print, lam_cvae: float = 0.):
        super().__init__()

        encoder_block_type = encoder_block_type.lower()
        if encoder_block_type == 'basic':
            encoder_block_class = BasicBlock
            model_arch = 'resnet34'
        elif encoder_block_type == 'bottleneck':
            encoder_block_class = Bottleneck
            model_arch = 'resnet50'
        else:
            raise NameError(encoder_block_type)

        self.frame_encoder = FrameEncoder(encoder_block_class, [3, 4, 6, 3], img_dim=img_dim * inp_frm)

        assert isinstance(bgd_encoder, BackgroundEncoder), f"{type(bgd_encoder)}"
        self.bgd_encoder = bgd_encoder

        assert lam_cvae in (0., 1.), "Only support `lam_cvae = 0 or 1` now."
        self.use_cvae = bool(int(lam_cvae))
        if self.use_cvae:
            self.cvae_32 = CVAE(32**2, 256, 2, True, self.bgd_encoder.num_classes)
            self.cvae_64 = CVAE(64**2, 256, 2, True, self.bgd_encoder.num_classes)

        self.frame_decoder = FrameDecoder(encoder_block_type, decoder_block_type, num_frm=tgt_frm)

        if pretrained_encoder:
            writer(f"Loading pre-trained weights on ImageNet for encoder {model_arch} ...")
            state_dict = load_state_dict_from_url(model_urls[model_arch])
            _ret_load_info = self.encoder.load_state_dict(state_dict, strict=False)
            writer(f"missing_keys: {_ret_load_info[0]}")
            writer(f"unexpected_keys: {_ret_load_info[1]}")
        else:
            writer(f"Do NOT load the pre-trained weights on ImageNet for encoder {model_arch}")

    def forward(self, inp_snp: Tensor, inp_bgd, sce_cls=None):
        # inp_snp: [b, c, t, h, w], inp_bgd: [b, c, h, w]
        assert inp_snp.ndim == 5, f"{inp_snp.shape}"
        b, c, t, h, w = inp_snp.shape
        inp_snp = rearrange(inp_snp, 'b c t h w -> b (c t) h w')

        feat_list: List[Tensor] = self.frame_encoder(inp_snp)  # (c',h'w')=(64,128X),(256|64,64**2),(512|128,32**2),#(1024,16),(2048,8); #(2048,2),(8192,)
        # feat_list[i]: [b, c', h', w']

        if self.use_cvae:
            if sce_cls:
                sce_cls = torch.as_tensor([sce_cls] * b, dtype=torch.int64, device=inp_snp.device)
            else:
                sce_cls: Tensor = self.bgd_encoder(inp_bgd, False)  # [b, d]
                sce_cls = torch.argmax(sce_cls, 1, True)

            c = feat_list[1].shape[1]
            feat_32: Tensor = rearrange(feat_list[1], 'b c h w -> (b c) (h w)')  # [b, c, h*w] = [b, 128, 32*32]
            rec_x32, mean_32, log_var_32, z_32 = self.cvae_32(feat_32.detach(), torch.repeat_interleave(sce_cls, c, 0))
            feat_list[1] = rearrange(rec_x32, '(b c) (h w) -> b c h w', b=b, c=128, h=32, w=32)

            c = feat_list[0].shape[1]
            feat_64: Tensor = rearrange(feat_list[0], 'b c h w -> (b c) (h w)')  # [b, c, h*w] = [b, 64, 64*64]
            rec_x64, mean_64, log_var_64, z_64 = self.cvae_64(feat_64.detach(), torch.repeat_interleave(sce_cls, c, 0))
            feat_list[0] = rearrange(rec_x64, '(b c) (h w) -> b c h w', b=b, c=64, h=64, w=64)

        rec_img = self.frame_decoder(feat_list)  # [b, c, t, h, w]

        if self.use_cvae:
            return rec_img, {'rec_x32': rec_x32, 'feat_32': feat_32.detach(), 'mean_32': mean_32, 'log_var_32': log_var_32}, {'rec_x64': rec_x64, 'feat_64': feat_64.detach(), 'mean_64': mean_64, 'log_var_64': log_var_64}
        else:
            return rec_img


class BidirectionalFrameAE(nn.Module):
    def __init__(self, img_dim: int = 3, inp_frm: int = 8, tgt_frm: int = 1, scene_classes=2, lam_cvae=1., encoder_block_type: str = 'basic', decoder_block_type='basic'):
        super().__init__()
        f_bgd_encoder = BackgroundEncoder(scene_classes)
        b_bgd_encoder = BackgroundEncoder(scene_classes)
        self.f_frameAE = SceneFrameAE(img_dim, inp_frm, tgt_frm, f_bgd_encoder, encoder_block_type, decoder_block_type, lam_cvae=lam_cvae)
        self.b_frameAE = SceneFrameAE(img_dim, inp_frm, 1, b_bgd_encoder, encoder_block_type, decoder_block_type, lam_cvae=lam_cvae)

    def forward(self, inp_frm: Tensor):
        # inp_frm: [b, c, t, h, w]
        # pred_forward_frm = self.f_frameAE(inp_frm)  # [b, c, tgt_frm+pre_frm, h, w]
        # pre_backward_frm = self.b_frameAE(torch.cat([pred_forward_frm, torch.flip(inp_frm[:, :, 1:], [-1])], -1))
        # return pred_forward_frm, pre_backward_frm
        raise NotImplementedError()
