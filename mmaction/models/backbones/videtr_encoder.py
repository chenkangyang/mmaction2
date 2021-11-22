import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcls.models.backbones import VisionTransformer
from mmcls.models.backbones.vision_transformer import HybridEmbed, PatchEmbed
from mmcv import ConfigDict
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import _load_checkpoint, load_state_dict
from mmcv.utils import to_2tuple
from mmdet.models.builder import BACKBONES as MMDET_BACKBONES
from mmdet.models.builder import build_backbone as build_mmdet_backbone

from mmaction.models.backbones import TimeSformer
from ...utils import get_root_logger
from ..builder import BACKBONES as MMACTION_BACKBONES
from ..builder import VIDETR_ACTOR_ENCODER
from ..builder import build_backbone as build_mmaction_backbone
from ..builder import build_videtr_actor_encoder


class VitDetPatchEmbed(PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, conv_cfg=None):
        super().__init__(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim, conv_cfg=conv_cfg)
    
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't " \
        #     f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

@MMDET_BACKBONES.register_module()
class ViTDetEncoder(VisionTransformer):

    def __init__(self,
                 img_size,
                 num_layers=12,
                 embed_dim=768,
                 num_heads=12,
                 patch_size=16,
                 in_channels=3,
                 feedforward_channels=3072,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 hybrid_backbone=None,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2,
                 det_token_num=100,
                 det_img_size=(800, 1344),
                 mid_pe_size=None,
                 use_checkpoint=False,
                 pretrained=None):
        super().__init__(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            hybrid_backbone=hybrid_backbone,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            num_fcs=num_fcs)
        
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.pretrained = pretrained
        img_size = to_2tuple(img_size)
        
        # set finetune flag
        self.has_mid_pe = False
        
        self.patch_embed = VitDetPatchEmbed(
            img_size=det_img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim)
        
        import math
        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(
            torch.zeros(1, det_token_num, self.embed_dim))

        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]

        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)

        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)

        B, E, Q = patch_pos_embed.shape
        P_H, P_W = img_size[0] // patch_size, img_size[
            1] // patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = det_img_size
        new_P_H, new_P_W = H // patch_size, W // patch_size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(
            torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1))

        self.img_size = det_img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print('No mid pe')
        else:
            print('Has mid pe')
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    num_layers - 1, 1, 1 +
                    (mid_pe_size[0] * mid_pe_size[1] // patch_size**2) + 100,
                    self.embed_dim))
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        
        self.use_checkpoint = use_checkpoint
    
    #TODO pretrain: 使用YOLOS的COCO预训练参数，模仿timesformer.init_weights写法
    def init_weights(self, pretrained=None):
        super(ViTDetEncoder, self).init_weights()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.det_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        if self.mid_pe_size:
            trunc_normal_(self.mid_pos_embed, std=.02)
        
        if pretrained:
            self.pretrained = pretrained
        
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
        
            state_dict = _load_checkpoint(self.pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            print("TODO: 加载Encoder的COCO预训练参数; mmcv._load_checkpoint成功，ckpt['state'] 结果如下", state_dict)
            #TODO 匹配参数，再加载进 self-->ViTDetEncoder
            # load_state_dict(self, state_dict, strict=False, logger=logger)
        
    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[
            1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat(
            (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed

    def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.det_token_num:, :]
        patch_pos_embed = pos_embed[:, :, 1:-self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        D, B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.mid_pe_size[0] // self.patch_size, self.mid_pe_size[
            1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(D * B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode='bicubic',
            align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(
            1, 2).contiguous().view(D, B, new_P_H * new_P_W, E)
        scale_pos_embed = torch.cat(
            (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W))
        else:
            temp_pos_embed = self.pos_embed

        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 -
                    self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W))
            else:
                temp_mid_pos_embed = self.mid_pos_embed


        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if self.has_mid_pe:
                if i < (self.num_layers - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm1(x)
        return x[:, -self.det_token_num:, :]
        

@MMACTION_BACKBONES.register_module()
class TimesFormerEncoder(TimeSformer):

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_ratio=0.,
                 transformer_layers=None,
                 attention_type='divided_space_time',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 **kwargs):
        super().__init__(
            num_frames,
            img_size,
            patch_size,
            pretrained=pretrained,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            in_channels=in_channels,
            dropout_ratio=dropout_ratio,
            transformer_layers=transformer_layers,
            attention_type=attention_type,
            norm_cfg=norm_cfg,
            **kwargs)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # x [batch_size * num_frames, num_patches, embed_dims]
        batches = x.shape[0]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        if self.attention_type != 'space_only':
            # x [batch_size, num_patches * num_frames + 1, embed_dims]
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_layers(x, None, None)

        if self.attention_type == 'space_only':
            # x [batch_size, num_patches + 1, embed_dims]
            x = x.view(-1, self.num_frames, *x.size()[-2:])
            x = torch.mean(x, 1)

        x = self.norm(x)

        return x[:, 1:]


@VIDETR_ACTOR_ENCODER.register_module()
class ACFormer(nn.Module):

    def __init__(self,
                 embed_dims=768,
                 num_heads=12,
                 transformer_layers=None,
                 num_transformer_layers=12,
                 num_actor_token = 100,
                 ):
        super().__init__()
        assert transformer_layers is None or isinstance(
            transformer_layers, (dict, list))
        self.num_transformer_layers = num_transformer_layers
        self.num_actor_token = num_actor_token
        self.actor_token = nn.Parameter(
            torch.zeros(1, num_actor_token, embed_dims)
        )
        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)
            _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=True,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]))
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i])),
                        operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                        norm_cfg=dict(type='LN', eps=1e-6),
                        batch_first=True)
                    for i in range(num_transformer_layers)
                ]
            
            transformer_layers = ConfigDict(
                dict(
                    type='TransformerLayerSequence',
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=num_transformer_layers))

        self.transformer_layers = build_transformer_layer_sequence(
            transformer_layers)

    def forward(self,
                keyframe_obj_token,
                clip_context_token,
                ):
        """Forward function for `ACFormer`.
        self-attention between all the tokens
        Args:
            keyframe_obj_token (Tensor): Input object detection token with shape
                `(num_queries, bs, embed_dims)`.
            clip_context_token (Tensor): Input space-time context token with shape
                `(num_queries, bs, embed_dims)`.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        B = clip_context_token.shape[0]
        actor_token = self.actor_token.expand(B, -1, -1)
        
        x = torch.cat((actor_token, keyframe_obj_token, clip_context_token), dim=1)
        x = self.transformer_layers(x, None, None)
        return x[:, :self.num_actor_token, :]


@MMACTION_BACKBONES.register_module()
class ViDETREncoder(nn.Module):

    def __init__(self, det_encoder, spacetime_encoder, actor_encoder):
        super().__init__()
        self.det_encoder = build_mmdet_backbone(det_encoder)
        self.spacetime_encoder = build_mmaction_backbone(spacetime_encoder)
        self.actor_encoder = build_videtr_actor_encoder(actor_encoder)
    
    def init_weights(self):
        pass

    def forward(self, clip, keyframe):
        #TODO 维度检查安排一下
        obj_token = self.det_encoder(keyframe)
        context_token = self.spacetime_encoder(clip)
        actor_token = self.actor_encoder(obj_token, context_token)
        return obj_token, actor_token 
