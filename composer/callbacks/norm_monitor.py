# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor weight and grad norms during training."""

from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger

__all__ = ['NormMonitor']


class NormMonitor(Callback): # MLPerf hacking
    """ TODO: change docstring
    Computes and logs the L2 norm of weights and grads on the :attr:`.Event.AFTER_TRAIN_BATCH` event.
    Args:
        log_layer_weight_norms (bool, optional): Whether to log the L2 normalization of each layer.
            Default: ``False``.
        
        log_layer_grad_norms (bool, optional): Whether to log the L2 normalization of each layer.
            Default: ``False``.

        log_layer_ratio_weight_grad_norms (bool, optional): Whether to log the ratio of L2 norms of weights and grads of each layer.
            Default: ``False``.

        eps (float, optional): Epsilon used in the denominator calculating the ratio of of L2 norms of weights and grads of each layer to avoid dividing by zero.
            Default: ``0.0001``.
    """

    def __init__(self, log_layer_weight_norms: bool = False, log_layer_grad_norms: bool = False, log_layer_ratio_weight_grad_norms = False, eps = 0.0001):
        self.log_layer_weight_norms = log_layer_weight_norms
        self.log_layer_grad_norms = log_layer_grad_norms
        self.log_layer_ratio_weight_grad_norms = log_layer_ratio_weight_grad_norms
        self.eps = eps

    def after_train_batch(self, state: State, logger: Logger):
        weight_norm = 0.0
        grad_norm = 0.0
        layer_weight_norms = {}
        layer_grad_norms = {}
        layer_ratio_weight_grad_norms = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_weight_norm = p.detach().data.norm(2).item()
                param_grad_norm = p.grad.detach().data.norm(2).item()
                param_ratio_weight_grad_norms = param_weight_norm / (param_grad_norm + self.eps)

                if self.log_layer_weight_norms:
                    layer_weight_norms[f'layer_weight_l2_norm/{name}'] = param_weight_norm

                if self.log_layer_grad_norms:
                    layer_grad_norms[f'layer_grad_l2_norm/{name}'] = param_grad_norm

                if self.log_layer_ratio_weight_grad_norms:
                    layer_ratio_weight_grad_norms[f'layer_ratio_weight_grad_l2_norm/{name}'] = param_ratio_weight_grad_norms

                param_weight_norm = param_weight_norm**2
                weight_norm += param_weight_norm

                param_grad_norm = param_grad_norm**2
                grad_norm += param_grad_norm

        weight_norm = weight_norm**0.5
        grad_norm = grad_norm**0.5
        logger.data_batch({'weight_l2_norm/step': weight_norm})
        logger.data_batch({'grad_l2_norm/step': grad_norm})
        # Would it make sense here to also log the general ratio of weight and grad norms across the network, not just by layer?
        if self.log_layer_weight_norms:
            logger.data_batch(layer_weight_norms)
        if self.log_layer_grad_norms:
            logger.data_batch(layer_grad_norms)
        if self.log_layer_ratio_weight_grad_norms:
            logger.data_batch(layer_ratio_weight_grad_norms)

