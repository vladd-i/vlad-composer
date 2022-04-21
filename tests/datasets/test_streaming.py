import tempfile
import time

import pytest
import torch

from composer.datasets.ade20k import RandomCropPair, StreamingADE20k
from composer.datasets.utils import NormalizationFn, pil_image_collate


@pytest.mark.timeout(100)
@pytest.mark.parametrize("protocol", ["s3", "sftp"])
@pytest.mark.parametrize("num_workers", [0, 1, 8])
def test_download_speed(protocol, num_workers):
    protocol_remote_dict = {
        "s3":
            "s3://mosaicml-internal-dataset-ade20k/mds/val",
        "sftp":
            "sftp://s-d26bfe922c2141cca.server.transfer.us-west-2.amazonaws.com/mosaicml-internal-dataset-ade20k/mds/val",
    }
    remote = protocol_remote_dict[protocol]
    tmpdir = tempfile.TemporaryDirectory()
    local = tmpdir.name
    device_batch_size = 32
    both_transform = RandomCropPair((32, 32))
    dataset = StreamingADE20k(remote=remote,
                              local=local,
                              shuffle=False,
                              device_batch_size=device_batch_size,
                              both_transform=both_transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=device_batch_size,
                                             num_workers=num_workers,
                                             drop_last=False,
                                             collate_fn=pil_image_collate)
    print(f"Init dataloader, remote={remote}, local={local}, expected_n_samples={dataset.index.total_samples}")

    count = 0
    start = time.time()
    for batch in dataloader:
        bs = batch[0].shape[0]
        count += bs
        print(count)
    end = time.time()

    duration = end - start
    print(f"n_samples={count}, download_time={duration} s")

    assert duration < 30, f"Dataset download was too slow ({duration} sec)"
