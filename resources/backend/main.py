# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requifFred by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import random

import numpy as np
import paddle

from resources.backend.paddlevideo.tasks import (test_model, train_dali, train_model,
                                                 train_model_multigrid)
from resources.backend.paddlevideo.utils import get_config, get_dist_info


def main():
    cfg = get_config('/home/lc/manet/save_step_80000/save_step_80000.pdparams', overrides=args.override)

    _, world_size = get_dist_info()
    parallel = world_size != 1
    if parallel:
        paddle.distributed.init_parallel_env()

    train_model(cfg,
                weights='/home/lc/manet/save_step_80000/save_step_80000.pdparams',
                # weights=args.weights,
                parallel=parallel,
                validate=args.validate,
                use_fleet=args.fleet,
                amp=args.amp,
                profiler_options=args.profiler_options)


if __name__ == '__main__':
    main()
