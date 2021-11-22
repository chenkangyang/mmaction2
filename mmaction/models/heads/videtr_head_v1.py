# # Copyright (c) OpenMMLab. All rights reserved.
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# from abc import ABCMeta, abstractmethod
# from mmcls.models.utils import to_2tuple
# from mmcv.cnn import Conv2d, Linear, build_activation_layer
# from mmcv.cnn.bricks.registry import NORM_LAYERS
# from mmcv.cnn.bricks.transformer import build_feedforward_network
# from mmcv.runner import force_fp32
# from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
#                         build_assigner, build_sampler, multi_apply,
#                         reduce_mean)
# from mmdet.core.bbox import bbox2roi
# from mmdet.models import HEADS as MMDET_HEADS
# from mmdet.models import build_loss as build_mmdet_loss
# from timm.models.byoanet import eca_lambda_resnext26ts
# from timm.models.layers import DropPath, Mlp, trunc_normal_

# from mmaction.core.bbox import bbox2result
# from mmaction.utils import import_module_error_class
# from ..builder import HEADS, MODELS
# from ..detectors.misc import Block, HybridEmbed, PatchEmbed

# BBOX_OBJDET_ENCODER = MODELS
# BBOX_ACTORDET_DECODER = MODELS

# BBOX_OBJDET_BRANCH = MODELS
# BBOX_ACTORDET_BRANCH = MODELS


# def build_objdet_encoder(cfg):
#     return BBOX_OBJDET_ENCODER.build(cfg)


# def build_actordet_decoder(cfg):
#     return BBOX_ACTORDET_DECODER.build(cfg)


# def build_objdet_branch(cfg):
#     """Build object bbox branch in videtr's bbox head."""
#     return BBOX_OBJDET_BRANCH.build(cfg)


# def build_actordet_branch(cfg):
#     """Build actor bbox branch in videtr's bbox head."""
#     return BBOX_ACTORDET_BRANCH.build(cfg)


# #********** 即插即用目标检测编码器 ObjectDetectionEncoder ***********#
# class ObjectDetectionEncoder(nn.Module, metaclass=ABCMeta):

#     def __init__(
#         self,
#         img_size,
#         patch_size=16,
#         det_token_num=100,
#         in_chans=3,
#         embed_dim=768,
#         depths=[12],
#         num_heads=[12],
#         mlp_ratio=4.,
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.,
#         norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()

#         if isinstance(img_size, tuple):
#             self.img_size = img_size
#         else:
#             self.img_size = to_2tuple(img_size)

#         self.patch_size = patch_size
#         self.det_token_num = det_token_num
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.depths = depths
#         self.num_layers = len(depths)
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.qkv_bias = qkv_bias
#         self.qk_scale = qk_scale
#         self.drop_rate = drop_rate
#         self.attn_drop_rate = attn_drop_rate
#         self.drop_path_rate = drop_path_rate
#         self.norm_layer = norm_layer

#     # @abstractmethod
#     # def init_weights(self):
#     #     """Initiate the parameters either from existing checkpoint or from
#     #     scratch."""

#     @abstractmethod
#     def forward(self, x):
#         """Defines the computation performed at every call."""

#     #TODO  返回 loss = self.loss_cls(初始化参数)
#     def loss(self, pred_coord, gt_coord, **kwargs):
#         pass


# @BBOX_OBJDET_ENCODER.register_module()
# class YOLOSObjectDetectionEncoder(ObjectDetectionEncoder):

#     def __init__(
#         self,
#         img_size,
#         init_pe_size=(800, 1344),
#         mid_pe_size=None,
#         patch_size=16,
#         det_token_num=100,
#         in_chans=3,
#         embed_dims=768,
#         depths=[12],
#         num_heads=[12],
#         mlp_ratio=4.,
#         qkv_bias=False,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.,
#         hybrid_backbone=None,
#         norm_layer=nn.LayerNorm,
#         is_distill=False,
#         use_checkpoint=False,
#     ):
#         super().__init__(
#             img_size=img_size,
#             patch_size=patch_size,
#             det_token_num=det_token_num,
#             in_chans=in_chans,
#             embed_dim=embed_dims,
#             depths=depths,
#             num_heads=num_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             drop_rate=drop_rate,
#             attn_drop_rate=attn_drop_rate,
#             drop_path_rate=drop_path_rate,
#         )
#         assert self.num_layers == 1
#         depth = depths[0]
#         num_heads = num_heads[0]

#         self.num_features = self.embed_dim = embed_dims  # num_features for consistency with other models
#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone,
#                 img_size=img_size,
#                 in_chans=in_chans,
#                 embed_dim=embed_dims)
#         else:
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size,
#                 patch_size=patch_size,
#                 in_chans=in_chans,
#                 embed_dim=embed_dims)
#         self.num_patches = self.patch_embed.num_patches
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
#         if is_distill:
#             self.pos_embed = nn.Parameter(
#                 torch.zeros(1, self.num_patches + 2, embed_dims))
#         else:
#             self.pos_embed = nn.Parameter(
#                 torch.zeros(1, self.num_patches + 1, embed_dims))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
#                ]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dims,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[i],
#                 norm_layer=norm_layer) for i in range(depth)
#         ])
#         self.norm = norm_layer(embed_dims)

#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#         # set finetune flag
#         self.has_mid_pe = False

#         self.finetune_det(
#             det_token_num=det_token_num,
#             img_size=init_pe_size,
#             mid_pe_size=mid_pe_size,
#             use_checkpoint=use_checkpoint)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def finetune_det(self,
#                      img_size=[800, 1344],
#                      det_token_num=100,
#                      mid_pe_size=None,
#                      use_checkpoint=False):

#         import math
#         g = math.pow(self.pos_embed.size(1) - 1, 0.5)
#         if int(g) - g != 0:
#             self.pos_embed = nn.Parameter(self.pos_embed[:, 1:, :])

#         self.det_token_num = det_token_num
#         self.det_token = nn.Parameter(
#             torch.zeros(1, det_token_num, self.embed_dim))
#         self.det_token = trunc_normal_(self.det_token, std=.02)
#         cls_pos_embed = self.pos_embed[:, 0, :]
#         cls_pos_embed = cls_pos_embed[:, None]
#         det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
#         det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
#         patch_pos_embed = self.pos_embed[:, 1:, :]
#         patch_pos_embed = patch_pos_embed.transpose(1, 2)
#         B, E, Q = patch_pos_embed.shape
#         P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[
#             1] // self.patch_size
#         patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
#         H, W = img_size
#         new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
#         patch_pos_embed = F.interpolate(
#             patch_pos_embed,
#             size=(new_P_H, new_P_W),
#             mode='bicubic',
#             align_corners=False)
#         patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
#         self.pos_embed = nn.Parameter(
#             torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1))
#         self.img_size = img_size
#         if mid_pe_size == None:
#             self.has_mid_pe = False
#             print('No mid pe')
#         else:
#             print('Has mid pe')
#             self.mid_pos_embed = nn.Parameter(
#                 torch.zeros(
#                     self.depth - 1, 1, 1 +
#                     (mid_pe_size[0] * mid_pe_size[1] // self.patch_size**2) +
#                     100, self.embed_dim))
#             trunc_normal_(self.mid_pos_embed, std=.02)
#             self.has_mid_pe = True
#             self.mid_pe_size = mid_pe_size
#         self.use_checkpoint = use_checkpoint

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'det_token'}

#     def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
#         # import pdb;pdb.set_trace()
#         cls_pos_embed = pos_embed[:, 0, :]
#         cls_pos_embed = cls_pos_embed[:, None]
#         det_pos_embed = pos_embed[:, -self.det_token_num:, :]
#         patch_pos_embed = pos_embed[:, 1:-self.det_token_num, :]
#         patch_pos_embed = patch_pos_embed.transpose(1, 2)
#         B, E, Q = patch_pos_embed.shape

#         P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[
#             1] // self.patch_size
#         patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

#         # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
#         # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

#         H, W = img_size
#         new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed,
#             size=(new_P_H, new_P_W),
#             mode='bicubic',
#             align_corners=False)
#         patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
#         scale_pos_embed = torch.cat(
#             (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
#         return scale_pos_embed

#     def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
#         # import pdb;pdb.set_trace()
#         cls_pos_embed = pos_embed[:, :, 0, :]
#         cls_pos_embed = cls_pos_embed[:, None]
#         det_pos_embed = pos_embed[:, :, -self.det_token_num:, :]
#         patch_pos_embed = pos_embed[:, :, 1:-self.det_token_num, :]
#         patch_pos_embed = patch_pos_embed.transpose(2, 3)
#         D, B, E, Q = patch_pos_embed.shape

#         P_H, P_W = self.mid_pe_size[0] // self.patch_size, self.mid_pe_size[
#             1] // self.patch_size
#         patch_pos_embed = patch_pos_embed.view(D * B, E, P_H, P_W)
#         H, W = img_size
#         new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
#         patch_pos_embed = nn.functional.interpolate(
#             patch_pos_embed,
#             size=(new_P_H, new_P_W),
#             mode='bicubic',
#             align_corners=False)
#         patch_pos_embed = patch_pos_embed.flatten(2).transpose(
#             1, 2).contiguous().view(D, B, new_P_H * new_P_W, E)
#         scale_pos_embed = torch.cat(
#             (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
#         return scale_pos_embed

#     def forward_features(self, x):
#         # import pdb;pdb.set_trace()
#         B, H, W = x.shape[0], x.shape[2], x.shape[3]

#         # if (H,W) != self.img_size:
#         #     self.finetune = True

#         x = self.patch_embed(x)
#         # interpolate init pe
#         if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
#             temp_pos_embed = self.InterpolateInitPosEmbed(
#                 self.pos_embed, img_size=(H, W))
#         else:
#             temp_pos_embed = self.pos_embed
#         # interpolate mid pe
#         if self.has_mid_pe:
#             # temp_mid_pos_embed = []
#             if (self.mid_pos_embed.shape[2] - 1 -
#                     self.det_token_num) != x.shape[1]:
#                 temp_mid_pos_embed = self.InterpolateMidPosEmbed(
#                     self.mid_pos_embed, img_size=(H, W))
#             else:
#                 temp_mid_pos_embed = self.mid_pos_embed

#         cls_tokens = self.cls_token.expand(
#             B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         det_token = self.det_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x, det_token), dim=1)
#         x = x + temp_pos_embed
#         x = self.pos_drop(x)

#         for i in range(len((self.blocks))):
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(self.blocks[i],
#                                           x)  # saves mem, takes time
#             else:
#                 x = self.blocks[i](x)
#             if self.has_mid_pe:
#                 if i < (self.depth - 1):
#                     x = x + temp_mid_pos_embed[i]

#         x = self.norm(x)

#         return x[:, -self.det_token_num:, :]

#     def forward_return_all_selfattention(self, x):
#         # import pdb;pdb.set_trace()
#         B, H, W = x.shape[0], x.shape[2], x.shape[3]

#         # if (H,W) != self.img_size:
#         #     self.finetune = True

#         x = self.patch_embed(x)
#         # interpolate init pe
#         if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
#             temp_pos_embed = self.InterpolateInitPosEmbed(
#                 self.pos_embed, img_size=(H, W))
#         else:
#             temp_pos_embed = self.pos_embed
#         # interpolate mid pe
#         if self.has_mid_pe:
#             # temp_mid_pos_embed = []
#             if (self.mid_pos_embed.shape[2] - 1 -
#                     self.det_token_num) != x.shape[1]:
#                 temp_mid_pos_embed = self.InterpolateMidPosEmbed(
#                     self.mid_pos_embed, img_size=(H, W))
#             else:
#                 temp_mid_pos_embed = self.mid_pos_embed

#         cls_tokens = self.cls_token.expand(
#             B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         det_token = self.det_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x, det_token), dim=1)
#         x = x + temp_pos_embed
#         x = self.pos_drop(x)
#         output = []
#         for i in range(len((self.blocks))):
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(self.blocks[i],
#                                           x)  # saves mem, takes time
#             else:
#                 x, attn = self.blocks[i](x, return_attention=True)

#             if i == len(self.blocks) - 1:
#                 output.append(attn)
#             if self.has_mid_pe:
#                 if i < (self.depth - 1):
#                     x = x + temp_mid_pos_embed[i]

#         x = self.norm(x)

#         return output

#     def forward(self, x, return_attention=False):
#         if return_attention == True:
#             # return self.forward_selfattention(x)
#             return self.forward_return_all_selfattention(x)
#         else:
#             x = self.forward_features(x)
#             return x


# @BBOX_OBJDET_ENCODER.register_module()
# class SwinYOLOSObjectDetectionEncoder(ObjectDetectionEncoder):
#     pass


# #********** 通用目标检测分支 ObjectDetectionBranch (静态物体类别+坐标) ***********#
# @BBOX_OBJDET_BRANCH.register_module()
# class ObjectDetectionBranch(nn.Module):
#     """
#     coord_encoder_cfg
#     reg_ffn (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for FFN, The order of the configs in the list should be
#             consistent with corresponding ffn in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config.
            
#     """

#     def __init__(
#         self,
#         obj_encoder_cfg,
#         reg_ffn,
#     ):
#         super().__init__()
#         self.coord_encoder = build_coord_encoder(coord_encoder_cfg)
#         self.reg_ffn = build_feedforward_network(reg_ffn) #TODO 接上两个头
#         self.coord_head = build_obj_det_head() #TODO 接上两个头
#         self.embed_dims = self.coord_encoder.embed_dims = self.coord_head.embed_dims

#     #TODO
#     def init_weights(self):
#         pass

#     def forward(self, x):  
#         #! 得到全部静态语义的tokens，
#         # 只有那些会动的tokens才对动作分类有用，
#         # 这个branch可以先筛出静态的人
#         x = self.coord_encoder(x)  # [bs, n, c]
#         all_obj_cls_score, all_obj_bbox_preds = self.coord_head(x)  # [bs, n, c]
#         return x

# #********** 即插即用的动作检测 Decoder ***********#
# from mmdet.models.utils.transformer import (TRANSFORMER_LAYER,
#                                             TRANSFORMER_LAYER_SEQUENCE,
#                                             build_transformer_layer_sequence)


# #! 需要暂时测试Decoder输入输出，后续需作为 Cls Branch 组件
# #********** 通用动作检测分支 ActorDetectionBranch (动态人体行为+坐标) ***********#
# @BBOX_ACTORDET_BRANCH.register_module()
# class ActorDetectionBranch(nn.Module):
#     #! 1.1 Object Tokens (Key1)
#     #! 1.2 Context Tokens (Key2)
#     #! 2. ActorTrack Tokens (Query)
#     #! (Key1 concat Key2) * (Query)
#     def __init__(
#         self,
#         actordet_decoder_cfg,
#         num_reg_fcs=2,
#         loss_cls=dict(
#             type='CrossEntropyLoss',
#             bg_cls_weight=0.1,
#             use_sigmoid=False,
#             loss_weight=1.0,
#             class_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=5.0),
#         loss_iou=dict(type='GIoULoss', loss_weight=2.0),
#         **kwargs
#     ):
#         super().__init__()
#         #TODO 加一个动态更新的query!
#         self.coord_encoder = build_actordet_decoder(actordet_decoder_cfg)
#         self.embed_dims = self.coord_encoder.embed_dims
        
#         self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)

#         self.loss_cls = build_loss(loss_cls)
#         self.loss_bbox = build_loss(loss_bbox)
#         self.loss_iou = build_loss(loss_iou)

#         # self.reg_ffn = build_feedforward_network(reg_ffn) #TODO 接上两个头
#         self.reg_ffn = FFN(
#             self.embed_dims,
#             self.embed_dims,
#             self.num_reg_fcs,
#             self.act_cfg,
#             dropout=0.0,
#             add_residual=False)
#         self.fc_reg = Linear(self.embed_dims, 4)

#     #TODO
#     def init_weights(self):
#         pass

#     def forward(self, x):  
#         #! 得到全部静态语义的tokens，
#         # 只有那些会动的tokens才对动作分类有用，
#         # 这个branch可以先筛出静态的人
#         outs_dec = self.coord_encoder(x)  # [bs, n, c]

#         all_cls_scores = self.fc_cls(outs_dec)
#         all_bbox_preds = self.fc_reg(self.activate(
#             self.reg_ffn(outs_dec))).sigmoid()
#         return all_actor_cls_scores, all_actor_bbox_preds

# @BBOX_ACTORDET_DECODER.register_module()
# class ActorContextDecoder(nn.Module, metaclass=ABCMeta):

#     def __init__(
#         self,
#         actor_embed_dims,
#         context_embed_dims,
#         decoder_cfg,
#     ):
#         super().__init__()
#         self.actor_embed_dims = actor_embed_dims
#         self.context_embed_dims = context_embed_dims
#         self.embed_dims = self.actor_embed_dims = self.context_embed_dims
#         self.decoder = build_transformer_layer_sequence(decoder_cfg)

#     #! 好ugly啊, 传入的actor_embeds是截断token数量后的@@@
#     def forward(self, actor_embeds, context_embeds, mask=None):
#         """Forward function for `ActorContextDecoder`.
#         Args:
#             actor_embeds (Tensor): The query embedding for encoder, with shape
#                 [bs, num_query, c].
            
#             context_embeds (Tensor): The key embedding for decoder, with shape
#                 [bs, num_query, c].
                
#             mask: mask matrix for cross-attention computing
#         Returns:
#             tuple[Tensor]: results of decoder containing the following tensor.
#                 - out_dec: Output from decoder. If return_intermediate_dec \
#                       is True output has shape [num_dec_layers, bs,
#                       num_query, embed_dims], else has shape [1, bs, \
#                       num_query, embed_dims].
#                 - context_memory: Output results from encoder, with shape \
#                       [bs, embed_dims, h, w].
#         """
#         bs, num_actor_tokens, embed_dims = actor_embeds.shape
#         _bs_context_tokens, num_context_tokens, _embed_dims = context_embeds.shape

#         # # use `view` instead of `flatten` for dynamically exporting to ONNX
#         actor_embeds = actor_embeds.view(bs, embed_dims, -1).permute(2, 0, 1)
#         context_embeds = context_embeds.view(bs, embed_dims,
#                                              -1).permute(2, 0, 1)

#         if mask is not None:
#             mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
#             out_dec = self.decoder(
#                 query=actor_embeds,
#                 key=context_embeds,
#                 value=context_embeds,
#                 key_padding_mask=mask,
#             )

#         out_dec = self.decoder(
#             query=actor_embeds,
#             key=context_embeds,
#             value=context_embeds,
#         )

#         out_dec = out_dec.transpose(1, 2)
#         return out_dec


# from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
# #TODO 继承自AnchorFreeHead，自己写一个详细的Head，要包含Loss的计算？参考 ACRNHead
# from mmdet.models.dense_heads import AnchorFreeHead


# @HEADS.register_module()
# class ViDETRHead(AnchorFreeHead):
#     """#TODO Implements the DETR transformer head.
#     See `paper: End-to-End Object Detection with Transformers
#     <https://arxiv.org/pdf/2005.12872>`_ for details.
#     Args:
#         num_classes (int): Number of categories excluding the background.
#         in_channels (int): Number of channels in the input feature map.
#         num_query (int): Number of query in Transformer.
#         num_reg_fcs (int, optional): Number of fully-connected layers used in
#             `FFN`, which is then used for the regression head. Default 2.
#         transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
#             Default: None.
#         sync_cls_avg_factor (bool): Whether to sync the avg_factor of
#             all ranks. Default to False.
#         positional_encoding (obj:`mmcv.ConfigDict`|dict):
#             Config for position encoding.
#         loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
#             classification loss. Default `CrossEntropyLoss`.
#         loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
#             regression loss. Default `L1Loss`.
#         loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
#             regression iou loss. Default `GIoULoss`.
#         tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
#             transformer head.
#         test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
#             transformer head.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Default: None
#     """

#     def __init__(
#             self,
#             num_classes,
#             obj_det_branch_cfg,
#             action_cls_branch_cfg,
#             loss_cls=dict(
#                 type='CrossEntropyLoss',
#                 bg_cls_weight=0.1,
#                 use_sigmoid=False,
#                 loss_weight=1.0,
#                 class_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=5.0),
#             loss_iou=dict(type='GIoULoss', loss_weight=2.0),
#             train_cfg=dict(
#                 assigner=dict(
#                     type='HungarianAssigner',
#                     cls_cost=dict(type='ClassificationCost', weight=1.),
#                     reg_cost=dict(type='BBoxL1Cost', weight=5.0),
#                     iou_cost=dict(type='IoUCost', iou_mode='giou',
#                                   weight=2.0))),
#             test_cfg=dict(max_per_img=100),
#             init_cfg=None,
#             **kwargs):
#         super(AnchorFreeHead, self).__init__(init_cfg)

#         class_weight = loss_cls.get('class_weight', None)
#         if class_weight is not None and (self.__class__ is ViDETRHead):
#             assert isinstance(class_weight, float), 'Expected ' \
#                 'class_weight to have type float. Found ' \
#                 f'{type(class_weight)}.'
#             # NOTE following the official DETR rep0, bg_cls_weight means
#             # relative classification weight of the no-object class.
#             bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
#             assert isinstance(bg_cls_weight, float), 'Expected ' \
#                 'bg_cls_weight to have type float. Found ' \
#                 f'{type(bg_cls_weight)}.'
#             class_weight = torch.ones(num_classes + 1) * class_weight
#             # set background class as the last indice
#             class_weight[num_classes] = bg_cls_weight
#             loss_cls.update({'class_weight': class_weight})
#             if 'bg_cls_weight' in loss_cls:
#                 loss_cls.pop('bg_cls_weight')
#             self.bg_cls_weight = bg_cls_weight

#         if train_cfg:
#             assert 'assigner' in train_cfg, 'assigner should be provided '\
#                 'when train_cfg is set.'
#             assigner = train_cfg['assigner']
#             assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
#                 'The classification weight for loss and matcher should be' \
#                 'exactly the same.'
#             assert loss_bbox['loss_weight'] == assigner['reg_cost'][
#                 'weight'], 'The regression L1 weight for loss and matcher ' \
#                 'should be exactly the same.'
#             assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
#                 'The regression iou weight for loss and matcher should be' \
#                 'exactly the same.'
#             self.assigner = build_assigner(assigner)
#             # DETR sampling=False, so use PseudoSampler
#             sampler_cfg = dict(type='PseudoSampler')
#             self.sampler = build_sampler(sampler_cfg, context=self)

#         self.num_classes = num_classes
#         self.fp16_enabled = False

#         self.loss_cls = build_mmdet_loss(loss_cls)
#         self.loss_bbox = build_mmdet_loss(loss_bbox)
#         self.loss_iou = build_mmdet_loss(loss_iou)

#         if self.loss_cls.use_sigmoid:
#             self.cls_out_channels = num_classes
#         else:
#             self.cls_out_channels = num_classes + 1

#         self.act_cfg = obj_det_branch_cfg.get('act_cfg',
#                                               dict(type='ReLU', inplace=True))
#         self.activate = build_activation_layer(self.act_cfg)
#         self.obj_coord_branch = build_obj_det_branch(obj_det_branch_cfg)
#         self.action_cls_branch = build_action_cls_branch(action_cls_branch_cfg)
#         self.embed_dims = self.obj_coord_branch.embed_dims = self.action_cls_branch.embed_dims

#         #TODO 建立与loss直接相连的层
#         self._init_layers()

#     #TODO 建立与loss直接相连的层
#     def _init_layers(self):
#         """Initialize layers of the transformer head."""
#         self.fc_obj_cls = Linear(self.embed_dims, self.cls_out_channels)
#         #! 已经在coord_branch中定义
#         # self.reg_ffn = FFN(
#         #     self.embed_dims,
#         #     self.embed_dims,
#         #     self.num_reg_fcs,
#         #     self.act_cfg,
#         #     dropout=0.0,
#         #     add_residual=False)
#         self.fc_obj_reg = Linear(self.embed_dims, 4)
#         #! 已经在coord_branch中定义

#     # TODO
#     def init_weights(self):
#         """Initialize weights of the transformer head."""
#         # The initialization for transformer is important
#         self.coord_branch.init_weights()
#         self.cls_branch.init_weights()

#     #TODO
#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         """load checkpoints."""
#         # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
#         # since `AnchorFreeHead._load_from_state_dict` should not be
#         # called here. Invoking the default `Module._load_from_state_dict`
#         # is enough.

#         # Names of some parameters in has been changed.
#         version = local_metadata.get('version', None)
#         if (version is None or version < 2) and self.__class__ is ViDETRHead:
#             convert_dict = {
#                 '.self_attn.': '.attentions.0.',
#                 '.ffn.': '.ffns.0.',
#                 '.multihead_attn.': '.attentions.1.',
#                 '.decoder.norm.': '.decoder.post_norm.'
#             }
#             state_dict_keys = list(state_dict.keys())
#             for k in state_dict_keys:
#                 for ori_key, convert_key in convert_dict.items():
#                     if ori_key in k:
#                         convert_key = k.replace(ori_key, convert_key)
#                         state_dict[convert_key] = state_dict[k]
#                         del state_dict[k]

#         super(AnchorFreeHead,
#               self)._load_from_state_dict(state_dict, prefix, local_metadata,
#                                           strict, missing_keys,
#                                           unexpected_keys, error_msgs)
#         pass

#     def forward(self, clip_feats, keyframe, clip_metas):
#         """Forward function.
#         Args:
#             clip_feats (tuple[Tensor]): Features from the upstream network, each is
#                 a 4D-tensor.
#             keyframe ([Tensor]): Original keyframe of the input, shape [bs, c, h, w]
#             clip_metas (list[dict]): List of clip information.
#         Returns:
#             tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.
#                 - all_cls_scores_list (list[Tensor]): Classification scores \
#                     for each scale level. Each is a 4D-tensor with shape \
#                     [nb_dec, bs, num_query, cls_out_channels]. Note \
#                     `cls_out_channels` should includes background.
#                 - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
#                     outputs for each scale level. Each is a 4D-tensor with \
#                     normalized coordinate format (cx, cy, w, h) and shape \
#                     [nb_dec, bs, num_query, 4].
#         """
#         num_levels = len(clip_feats)
#         clip_metas_list = [clip_metas for _ in range(num_levels)]
#         return multi_apply(self.forward_single, clip_feats, keyframe,
#                            clip_metas_list)

#     def forward_single(self, clip_feats, keyframe, clip_metas):
#         """"Forward function for a single feature level.
#         Args:
#             clip_feats (Tensor): Input feature from backbone's single stage, shape
#                 [bs, n, c].
#             keyframe (Tensor): Original keyframe of the input, shape [bs, c, h, w]
#             clip_metas (list[dict]): List of clip information.
#         Returns:
#             all_cls_scores (Tensor): Outputs from the classification head,
#                 shape [nb_dec, bs, num_query, cls_out_channels]. Note
#                 cls_out_channels should includes background.
#             all_bbox_preds (Tensor): Sigmoid outputs from the regression
#                 head with normalized coordinate format (cx, cy, w, h).
#                 Shape [nb_dec, bs, num_query, 4].
#         """
#         # construct binary masks which used for the transformer.
#         # NOTE following the official DETR repo, non-zero values representing
#         # ignored positions, while zero values means valid positions.
#         batch_size = clip_feats.size(0)
#         actor_embeds = self.coord_branch(keyframe)  # [bs, num_query, embed_dim]
        
#         # all_obj_cls_scores = self.fc_obj_cls()
#         # all_obj_cls_scores = self.fc_obj_cls(outs_dec)
#         # all_bbox_preds = self.fc_obj_reg(self.activate(
#         #     self.reg_ffn(outs_dec))).sigmoid()
        
        
#         for img_id in range(batch_size):
#             clip_metas[img_id]['clip_feats_shape'] = clip_feats[img_id].shape
#         context_embeds = clip_feats.flatten(2).transpose(
#             1, 2)  # [bs, embed_dim, t, h, w] --> [bs, t*h*w, embed_dim]
#         # TODO 20211020 actor_embeds 截断？
#         out_cls = self.cls_branch(actor_embeds, context_embeds)

#         # all_cls_scores = self.fc_cls(outs_dec)
#         # all_bbox_preds = self.fc_reg(self.activate(
#         #     self.reg_ffn(outs_dec))).sigmoid()
#         # return all_cls_scores, all_bbox_preds

#     #TODO
#     @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
#     def get_bboxes(self,
#                    all_cls_scores_list,
#                    all_bbox_preds_list,
#                    img_metas,
#                    rescale=False):
#         """Transform network outputs for a batch into bbox predictions.

#         Args:
#             all_cls_scores_list (list[Tensor]): Classification outputs
#                 for each feature level. Each is a 4D-tensor with shape
#                 [nb_dec, bs, num_query, cls_out_channels].
#             all_bbox_preds_list (list[Tensor]): Sigmoid regression
#                 outputs for each feature level. Each is a 4D-tensor with
#                 normalized coordinate format (cx, cy, w, h) and shape
#                 [nb_dec, bs, num_query, 4].
#             img_metas (list[dict]): Meta information of each image.
#             rescale (bool, optional): If True, return boxes in original
#                 image space. Default False.

#         Returns:
#             list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
#                 The first item is an (n, 5) tensor, where the first 4 columns \
#                 are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
#                 5-th column is a score between 0 and 1. The second item is a \
#                 (n,) tensor where each item is the predicted class label of \
#                 the corresponding box.
#         """
#         # NOTE defaultly only using outputs from the last feature level,
#         # and only the outputs from the last decoder layer is used.
#         cls_scores = all_cls_scores_list[-1][-1]
#         bbox_preds = all_bbox_preds_list[-1][-1]

#         result_list = []
#         for img_id in range(len(img_metas)):
#             cls_score = cls_scores[img_id]
#             bbox_pred = bbox_preds[img_id]
#             img_shape = img_metas[img_id]['img_shape']
#             scale_factor = img_metas[img_id]['scale_factor']
#             proposals = self._get_bboxes_single(cls_score, bbox_pred,
#                                                 img_shape, scale_factor,
#                                                 rescale)
#             result_list.append(proposals)

#         return result_list

#     #TODO
#     def _get_bboxes_single(self,
#                            cls_score,
#                            bbox_pred,
#                            img_shape,
#                            scale_factor,
#                            rescale=False):
#         """Transform outputs from the last decoder layer into bbox predictions
#         for each image.

#         Args:
#             cls_score (Tensor): Box score logits from the last decoder layer
#                 for each image. Shape [num_query, cls_out_channels].
#             bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
#                 for each image, with coordinate format (cx, cy, w, h) and
#                 shape [num_query, 4].
#             img_shape (tuple[int]): Shape of input image, (height, width, 3).
#             scale_factor (ndarray, optional): Scale factor of the image arange
#                 as (w_scale, h_scale, w_scale, h_scale).
#             rescale (bool, optional): If True, return boxes in original image
#                 space. Default False.

#         Returns:
#             tuple[Tensor]: Results of detected bboxes and labels.

#                 - det_bboxes: Predicted bboxes with shape [num_query, 5], \
#                     where the first 4 columns are bounding box positions \
#                     (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
#                     between 0 and 1.
#                 - det_labels: Predicted labels of the corresponding box with \
#                     shape [num_query].
#         """
#         assert len(cls_score) == len(bbox_pred)
#         max_per_img = self.test_cfg.get('max_per_img', self.num_query)
#         # exclude background
#         if self.loss_cls.use_sigmoid:
#             cls_score = cls_score.sigmoid()
#             scores, indexes = cls_score.view(-1).topk(max_per_img)
#             det_labels = indexes % self.num_classes
#             bbox_index = indexes // self.num_classes
#             bbox_pred = bbox_pred[bbox_index]
#         else:
#             scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
#             scores, bbox_index = scores.topk(max_per_img)
#             bbox_pred = bbox_pred[bbox_index]
#             det_labels = det_labels[bbox_index]

#         det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
#         det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
#         det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
#         det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
#         det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
#         if rescale:
#             det_bboxes /= det_bboxes.new_tensor(scale_factor)
#         det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

#         return det_bboxes, det_labels

#     #TODO 多头输出重组loss.
#     def loss(self,
#              all_cls_scores_list,
#              all_bbox_preds_list,
#              gt_bboxes_list,
#              gt_labels_list,
#              img_metas,
#              gt_bboxes_ignore=None):
#         """"Loss function.

#         Only outputs from the last feature level are used for computing
#         losses by default.

#         Args:
#             all_cls_scores_list (list[Tensor]): Classification outputs
#                 for each feature level. Each is a 4D-tensor with shape
#                 [nb_dec, bs, num_query, cls_out_channels].
#             all_bbox_preds_list (list[Tensor]): Sigmoid regression
#                 outputs for each feature level. Each is a 4D-tensor with
#                 normalized coordinate format (cx, cy, w, h) and shape
#                 [nb_dec, bs, num_query, 4].
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             img_metas (list[dict]): List of image meta information.
#             gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
#                 which can be ignored for each image. Default None.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         # NOTE defaultly only the outputs from the last feature scale is used.
#         all_cls_scores = all_cls_scores_list[-1]
#         all_bbox_preds = all_bbox_preds_list[-1]
#         assert gt_bboxes_ignore is None, \
#             'Only supports for gt_bboxes_ignore setting to None.'

#         num_dec_layers = len(all_cls_scores)
#         all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_bboxes_ignore_list = [
#             gt_bboxes_ignore for _ in range(num_dec_layers)
#         ]
#         img_metas_list = [img_metas for _ in range(num_dec_layers)]

#         losses_cls, losses_bbox, losses_iou = multi_apply(
#             self.loss_single, all_cls_scores, all_bbox_preds,
#             all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
#             all_gt_bboxes_ignore_list)

#         loss_dict = dict()
#         # loss from the last decoder layer
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_bbox'] = losses_bbox[-1]
#         loss_dict['loss_iou'] = losses_iou[-1]
#         # loss from other decoder layers
#         num_dec_layer = 0
#         for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
#                                                        losses_bbox[:-1],
#                                                        losses_iou[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
#             loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
#             num_dec_layer += 1
#         return loss_dict

#     #TODO 多头输出重组loss.
#     def loss_single(self,
#                     cls_scores,
#                     bbox_preds,
#                     gt_bboxes_list,
#                     gt_labels_list,
#                     img_metas,
#                     gt_bboxes_ignore_list=None):
#         """"Loss function for outputs from a single decoder layer of a single
#         feature level.

#         Args:
#             cls_scores (Tensor): Box score logits from a single decoder layer
#                 for all images. Shape [bs, num_query, cls_out_channels].
#             bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
#                 for all images, with normalized coordinate (cx, cy, w, h) and
#                 shape [bs, num_query, 4].
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             img_metas (list[dict]): List of image meta information.
#             gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                 boxes which can be ignored for each image. Default None.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components for outputs from
#                 a single decoder layer.
#         """
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
#         cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
#                                            gt_bboxes_list, gt_labels_list,
#                                            img_metas, gt_bboxes_ignore_list)
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          num_total_pos, num_total_neg) = cls_reg_targets
#         labels = torch.cat(labels_list, 0)
#         label_weights = torch.cat(label_weights_list, 0)
#         bbox_targets = torch.cat(bbox_targets_list, 0)
#         bbox_weights = torch.cat(bbox_weights_list, 0)

#         # classification loss
#         cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
#         # construct weighted avg_factor to match with the official DETR repo
#         cls_avg_factor = num_total_pos * 1.0 + \
#             num_total_neg * self.bg_cls_weight
#         if self.sync_cls_avg_factor:
#             cls_avg_factor = reduce_mean(
#                 cls_scores.new_tensor([cls_avg_factor]))
#         cls_avg_factor = max(cls_avg_factor, 1)

#         loss_cls = self.loss_cls(
#             cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

#         # Compute the average number of gt boxes accross all gpus, for
#         # normalization purposes
#         num_total_pos = loss_cls.new_tensor([num_total_pos])
#         num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

#         # construct factors used for rescale bboxes
#         factors = []
#         for img_meta, bbox_pred in zip(img_metas, bbox_preds):
#             img_h, img_w, _ = img_meta['img_shape']
#             factor = bbox_pred.new_tensor([img_w, img_h, img_w,
#                                            img_h]).unsqueeze(0).repeat(
#                                                bbox_pred.size(0), 1)
#             factors.append(factor)
#         factors = torch.cat(factors, 0)

#         # DETR regress the relative position of boxes (cxcywh) in the image,
#         # thus the learning target is normalized by the image size. So here
#         # we need to re-scale them for calculating IoU loss
#         bbox_preds = bbox_preds.reshape(-1, 4)
#         bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
#         bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

#         # regression IoU loss, defaultly GIoU loss
#         loss_iou = self.loss_iou(
#             bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

#         # regression L1 loss
#         loss_bbox = self.loss_bbox(
#             bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
#         return loss_cls, loss_bbox, loss_iou

#     #TODO
#     def get_targets(self,
#                     cls_scores_list,
#                     bbox_preds_list,
#                     gt_bboxes_list,
#                     gt_labels_list,
#                     img_metas,
#                     gt_bboxes_ignore_list=None):
#         """"Compute regression and classification targets for a batch image.

#         Outputs from a single decoder layer of a single feature level are used.

#         Args:
#             cls_scores_list (list[Tensor]): Box score logits from a single
#                 decoder layer for each image with shape [num_query,
#                 cls_out_channels].
#             bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
#                 decoder layer for each image, with normalized coordinate
#                 (cx, cy, w, h) and shape [num_query, 4].
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             img_metas (list[dict]): List of image meta information.
#             gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                 boxes which can be ignored for each image. Default None.

#         Returns:
#             tuple: a tuple containing the following targets.

#                 - labels_list (list[Tensor]): Labels for all images.
#                 - label_weights_list (list[Tensor]): Label weights for all \
#                     images.
#                 - bbox_targets_list (list[Tensor]): BBox targets for all \
#                     images.
#                 - bbox_weights_list (list[Tensor]): BBox weights for all \
#                     images.
#                 - num_total_pos (int): Number of positive samples in all \
#                     images.
#                 - num_total_neg (int): Number of negative samples in all \
#                     images.
#         """
#         assert gt_bboxes_ignore_list is None, \
#             'Only supports for gt_bboxes_ignore setting to None.'
#         num_imgs = len(cls_scores_list)
#         gt_bboxes_ignore_list = [
#             gt_bboxes_ignore_list for _ in range(num_imgs)
#         ]

#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          pos_inds_list,
#          neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
#                                       bbox_preds_list, gt_bboxes_list,
#                                       gt_labels_list, img_metas,
#                                       gt_bboxes_ignore_list)
#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, bbox_targets_list,
#                 bbox_weights_list, num_total_pos, num_total_neg)

#     #TODO
#     def _get_target_single(self,
#                            cls_score,
#                            bbox_pred,
#                            gt_bboxes,
#                            gt_labels,
#                            img_meta,
#                            gt_bboxes_ignore=None):
#         """"Compute regression and classification targets for one image.

#         Outputs from a single decoder layer of a single feature level are used.

#         Args:
#             cls_score (Tensor): Box score logits from a single decoder layer
#                 for one image. Shape [num_query, cls_out_channels].
#             bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
#                 for one image, with normalized coordinate (cx, cy, w, h) and
#                 shape [num_query, 4].
#             gt_bboxes (Tensor): Ground truth bboxes for one image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (Tensor): Ground truth class indices for one image
#                 with shape (num_gts, ).
#             img_meta (dict): Meta information for one image.
#             gt_bboxes_ignore (Tensor, optional): Bounding boxes
#                 which can be ignored. Default None.

#         Returns:
#             tuple[Tensor]: a tuple containing the following for one image.

#                 - labels (Tensor): Labels of each image.
#                 - label_weights (Tensor]): Label weights of each image.
#                 - bbox_targets (Tensor): BBox targets of each image.
#                 - bbox_weights (Tensor): BBox weights of each image.
#                 - pos_inds (Tensor): Sampled positive indices for each image.
#                 - neg_inds (Tensor): Sampled negative indices for each image.
#         """

#         num_bboxes = bbox_pred.size(0)
#         # assigner and sampler
#         assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
#                                              gt_labels, img_meta,
#                                              gt_bboxes_ignore)
#         sampling_result = self.sampler.sample(assign_result, bbox_pred,
#                                               gt_bboxes)
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds

#         # label targets
#         labels = gt_bboxes.new_full((num_bboxes, ),
#                                     self.num_classes,
#                                     dtype=torch.long)
#         labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
#         label_weights = gt_bboxes.new_ones(num_bboxes)

#         # bbox targets
#         bbox_targets = torch.zeros_like(bbox_pred)
#         bbox_weights = torch.zeros_like(bbox_pred)
#         bbox_weights[pos_inds] = 1.0
#         img_h, img_w, _ = img_meta['img_shape']

#         # DETR regress the relative position of boxes (cxcywh) in the image.
#         # Thus the learning target should be normalized by the image size, also
#         # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
#         factor = bbox_pred.new_tensor([img_w, img_h, img_w,
#                                        img_h]).unsqueeze(0)
#         pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
#         pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
#         bbox_targets[pos_inds] = pos_gt_bboxes_targets
#         return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
#                 neg_inds)

#     # over-write because img_metas are needed as inputs for bbox_head.
#     def forward_train(self,
#                       clip_feats,
#                       keyframe,
#                       clip_metas,
#                       gt_bboxes,
#                       gt_labels=None,
#                       gt_bboxes_ignore=None,
#                       proposal_cfg=None,
#                       **kwargs):
#         """Forward function for training mode.

#         Args:
#         #TODO 重写shape
#             clip_feats (list[Tensor]): Maybe multi level features from backbone of .
#             keyframe (Tensor): Input keyframe of shape (N, C, H, W).
#             clip_metas (list[dict]): Meta information of each clip, e.g.,
#                 clip space shape, scaling factor, etc.
#             gt_bboxes (Tensor): Ground truth bboxes of the image,
#                 shape (num_gts, 4).
#             gt_labels (Tensor): Ground truth labels of each box,
#                 shape (num_gts,).
#             gt_bboxes_ignore (Tensor): Ground truth bboxes to be
#                 ignored, shape (num_ignored_gts, 4).
#             proposal_cfg (mmcv.Config): Test / postprocessing configuration,
#                 if None, test_cfg would be used.

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         assert proposal_cfg is None, '"proposal_cfg" must be None'

#         if isinstance(clip_feats, torch.Tensor):
#             clip_feats = [clip_feats]

#         keyframe = [keyframe]

#         outs = self(clip_feats, keyframe, clip_metas)
#         if gt_labels is None:
#             loss_inputs = outs + (gt_bboxes, clip_metas)
#         else:
#             loss_inputs = outs + (gt_bboxes, gt_labels, clip_metas)
#         losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
#         return losses
