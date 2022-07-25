# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Wikipedia 2020-01-01 dataset.

"""
import copy
import logging
import os
from functools import partial
from itertools import chain, cycle
from typing import Any, Dict, Optional

from torch.utils.data import IterableDataset, get_worker_info

from composer.datasets.streaming import StreamingDataset
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['StreamingWiki']


class StreamingWiki(StreamingDataset):
    """
    Implementation of the Wikipedia 2020-01-01 dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        # split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Currently only supporting ``'truncate'``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def _decode(self, data: bytes) -> str:
        return data.decode('utf-8')

    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        else:
            truncation = False
            padding = False
            max_length = None
        return self.tokenizer(text_sample['text'], truncation=truncation, padding=padding, max_length=max_length)

    def __init__(self,
                 remote: str,
                 local: str,
                 # split: str,
                 shuffle: bool,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 batch_size: Optional[int] = None):

        # HF Transformers is needed to build the tokenizer
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        # Validation
        # if split not in ['train', 'val']:
        #     raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate']:
            raise ValueError(f"Only group_method='truncate' is supported at this time.")

        # Build StreamingDataset
        decoders = {
            'text': self._decode,
        }
        super().__init__(remote=remote,
                         local=local,
                         shuffle=shuffle,
                         decoders=decoders,
                         batch_size=batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        # Skip any token grouping, currently only supporting group_method='truncate'
        return token_sample
