# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor weights during training."""

from composer.core import State
from composer.core.callback import Callback
from composer.loggers import Logger

__all__ = ['WeightMonitor']


class WeightMonitor(Callback):
    """ TODO: change docstring
    Computes and logs the L2 norm of weights on the :attr:`.Event.AFTER_TRAIN_BATCH` event.

    L2 norms are calculated after the reduction of gradients across GPUs. This function iterates over the parameters of
    the model and may cause a reduction in throughput while training large models. In order to ensure the
    correctness of the norm, this function should be called after gradient unscaling in cases where gradients are scaled.

    Example:
    .. doctest::

        >>> from composer import Trainer
        >>> from composer.callbacks import WeightMonitor
        >>> # constructing trainer object with this callback
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     eval_dataloader=eval_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[WeightMonitor()],
        ... )

    The L2 norms are logged by the :class:`.Logger` to the following keys as described below.

    +-----------------------------------+-------------------------------------------------------------+
    | Key                               | Logged data                                                 |
    +===================================+=============================================================+
    |                                   | L2 norm of the weights of all parameters in the model     |
    | ``weight_l2_norm/step``             | on the :attr:`.Event.AFTER_TRAIN_BATCH` event.              |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+
    |                                   | Layer-wise L2 norms if ``log_layer_weight_norms``             |
    | ``layer_weight_l2_norm/LAYER_NAME`` | is ``True``. Default: ``False``.                            |
    |                                   |                                                             |
    +-----------------------------------+-------------------------------------------------------------+

    Args:
        log_layer_weight_norms (bool, optional): Whether to log the L2 normalization of each layer.
            Default: ``False``.
    """

    def __init__(self, log_layer_weight_norms: bool = False):
        self.log_layer_weight_norms = log_layer_weight_norms

    def after_train_batch(self, state: State, logger: Logger):
        norm = 0.0
        layer_norms = {}
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                param_weight_norm = p.detach().data.norm(2).item()  # type: ignore
                if self.log_layer_weight_norms:
                    layer_norms[f'layer_weight_l2_norm/{name}'] = param_weight_norm

                param_weight_norm = param_weight_norm**2
                norm += param_weight_norm

        norm = norm**0.5
        logger.data_batch({'weight_l2_norm/step': norm})
        if self.log_layer_weight_norms:
            logger.data_batch(layer_norms)
