# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fast_mertro import FastMETRO_Body_Network

__all__ = ['build_head']

def build_head(cfg):
    if cfg.MODEL.head == 'fast_metro':
        return FastMETRO_Body_Network(cfg)
    else:
        raise NotImplementedError(cfg.MODEL.head)

