# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hrnet import get_pose_net as get_hrnet
from resnet import get_pose_net as get_resnet

__all__ = ['build_encoder']

def build_encoder(cfg):
    if cfg.MODEL.encoder == 'hrnet':
        return get_hrnet(cfg)
    elif  cfg.MODEL.encoder == 'resnet':
        return get_resnet(cfg)
    else:
        raise NotImplementedError(cfg.MODEL.encoder)

