# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import signal
import os
import paddle
from paddle.fluid.dataloader import BatchSampler
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.vision import Compose

from .registry import DATASETS, PIPELINES, DATALOADERS, BATCH_SAMPLERS, SAMPLERS
from ..utils import get_logger
from ..utils.build_utils import build
import numpy as np

logger = get_logger("paddlevideo")


def build_pipeline(cfg):
    """Build pipeline.
    Args:
        cfg (dict): root config dict.
    """
    if cfg == None:
        return
    return Compose(cfg)


def build_dataset(cfg):
    """Build dataset.
    Args:
        cfg (dict): root config dict.

    Returns:
        dataset: dataset.
    """
    # XXX: ugly code here!
    cfg_dataset, cfg_pipeline = cfg
    cfg_dataset.pipeline = build_pipeline(cfg_pipeline)
    dataset = build(cfg_dataset, DATASETS, key="format")
    return dataset


def build_sampler(cfg):
    """Build batch_sampler.
    Args:
        cfg (dict): root config dict.

    Returns:
        batch_sampler: batch_sampler.
    """
    sampler = build(cfg, SAMPLERS)
    return sampler


def build_batch_pipeline(cfg):
    batch_pipeline = build(cfg, PIPELINES)
    return batch_pipeline


def build_custom_dataloader(cfg):
    custom_dataloader = build(cfg, DATALOADERS, key='dataloader')
    return custom_dataloader


def build_dataloader(dataset,
                     batch_size,
                     num_workers,
                     places=None,
                     shuffle=True,
                     drop_last=True,
                     multigrid=False,
                     collate_fn_cfg=None,
                     **kwargs):
    """Build Paddle Dataloader.

    XXX explain how the batch_sampler work!

    Args:
        dataset (paddle.dataset): A PaddlePaddle dataset object.
        batch_size (int): batch size on single card.
        num_worker (int): num_worker
        shuffle(bool): whether to shuffle the data at every epoch.
    """

    if not kwargs.get('sampler'):
        batch_sampler = DistributedBatchSampler(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    drop_last=drop_last)
    else:
        sampler = build_sampler(kwargs['sampler'])
        batch_sampler = BatchSampler(dataset,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     drop_last=drop_last)
    kwargs.update({'batch_sampler': batch_sampler})

    # NOTE(shipping): when switch the mix operator on, such as: mixup, cutmix.

    # batch like: [[img, label, attibute, ...], [imgs, label, attribute, ...], ...] will recollate to:
    # [[img, img, ...], [label, label, ...], [attribute, attribute, ...], ...] as using numpy.transpose.

    def mix_collate_fn(batch):
        pipeline = build_batch_pipeline(collate_fn_cfg)
        batch = pipeline(batch)
        slots = []
        for items in batch:
            for i, item in enumerate(items):
                if len(slots) < len(items):
                    slots.append([item])
                else:
                    slots[i].append(item)
        return [np.stack(slot, axis=0) for slot in slots]

    # if collate_fn_cfg is not None:
    # ugly code here. collate_fn is mix op config
    #    collate_fn = mix_collate_fn(collate_fn_cfg)

    data_loader = DataLoader(
        dataset,
        places=places,
        num_workers=num_workers,
        collate_fn=mix_collate_fn if collate_fn_cfg is not None else None,
        **kwargs)

    return data_loader


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    logger.info("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)
    return


signal.signal(signal.SIGINT, term_mp)
signal.signal(signal.SIGTERM, term_mp)
