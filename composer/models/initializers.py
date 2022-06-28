# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Module Initializers."""

from typing import Callable

import torch
from torch import nn as nn

import math

from composer.utils import StringEnum


class Initializer(StringEnum):
    """Sets the initialization scheme for different layers of a PyTorch model."""
    KAIMING_NORMAL = 'kaiming_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    BN_UNIFORM = 'bn_uniform'
    BN_ONES = 'bn_ones'
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    LINEAR_LOG_CONSTANT_BIAS = 'linear_log_constant_bias'
    FC_HACK = 'fc_hack'  # MLPerf hacking
    OTHER_HACK = 'other_hack' # MLPerf hacking

    def get_initializer(self) -> Callable[[torch.nn.Module], None]:
        """Get the initializer function.

        Returns:
            (torch.nn.Module) -> None: The initializer function.
        """

        def kaiming_normal(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(w.weight)

        def kaiming_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(w.weight)

        def xavier_uniform(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(w.weight)

        def xavier_normal(w: nn.Module):
            if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(w.weight)

        def bn_ones(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.ones_like(w.weight.data)
                w.bias.data = torch.zeros_like(w.bias.data)

        def bn_uniform(w: nn.Module):
            if isinstance(w, torch.nn.BatchNorm2d):
                w.weight.data = torch.rand(w.weight.data.shape)
                w.bias.data = torch.zeros_like(w.bias.data)

        def linear_log_constant_bias(w: nn.Module):
            if isinstance(w, torch.nn.Linear):
                w.bias.data = torch.ones(w.bias.shape) * -torch.log(torch.tensor(w.bias.shape[0]))

        def fc_hack(w: nn.Module):  # MLPerf hacking
            if isinstance(w, torch.nn.Linear):
                # initialize FC weights with Normal(0, 0.01)
                torch.nn.init.normal_(w.weight, mean=0, std=0.01)

                # initialize FC biases equivalently to NVIDIA's 
                # mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) 
                # assuming fan_in = 1 for bias
                if w.bias is not None:
                    std = math.sqrt(2.0)
                    torch.nn.init._no_grad_normal_(w.bias, 0., std)

        def other_hack(w: nn.Module): # MLPerf hacking
            if not isinstance(w, torch.nn.Linear):
                # initialize other layers' weights equivalently to NVIDIA's 
                # mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) 
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w.weight)
                std = math.sqrt(2.0 / float(fan_in))
                torch.nn.init._no_grad_normal_(w.weight, 0., std)

                # initialize other layers' biases equivalently to NVIDIA's 
                # mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) 
                # assuming fan_in = 1 for bias
                if w.bias is not None:
                    std = math.sqrt(2.0)
                    torch.nn.init._no_grad_normal_(w.bias, 0., std)

        initializer_dict = {
            'kaiming_normal': kaiming_normal,
            'kaiming_uniform': kaiming_uniform,
            'bn_uniform': bn_uniform,
            'bn_ones': bn_ones,
            'xavier_uniform': xavier_uniform,
            'xavier_normal': xavier_normal,
            'linear_log_constant_bias': linear_log_constant_bias,
            'fc_hack': fc_hack,
            'other_hack': other_hack
        }
        if self.value not in initializer_dict:
            raise ValueError(f"Initializer '{self.value}' not found.")
        return initializer_dict[self.value]
