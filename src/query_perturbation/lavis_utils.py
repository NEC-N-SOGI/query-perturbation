import argparse
import random
from pathlib import Path
from time import time
from typing import Optional

import lavis.tasks as tasks
import numpy as np
import torch
import torch.nn.functional as F
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.datasets.datasets.dataloader_utils import PrefetchLoader
from lavis.models.blip2_models.blip2 import LayerNorm as LavisLayerNorm
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.runners.runner_base import RunnerBase
from lavis.tasks.base_task import BaseTask
from torch.backends import cudnn
from torch.utils.data import DataLoader


def layer_norm_forward(self: LavisLayerNorm, x: torch.Tensor) -> torch.Tensor:
    _x: torch.Tensor = F.layer_norm(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )
    return _x


# Lavis's LayerNorm cast the input to float32.
# to avoid this, we need to override the forward method.
LavisLayerNorm.forward = layer_norm_forward


def setup_seeds(config: Config) -> None:
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args(config_path: str | Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg-path", required=True, help="path to configuration file."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args(args=["--cfg-path", str(config_path)])

    return args


def get_config(config_path: str | Path) -> Config:
    return Config(parse_args(str(config_path)))


def get_task(config_path: str | Path, cfg: Optional[Config] = None) -> BaseTask:
    if cfg is None:
        cfg = get_config(str(config_path))
    return cfg, tasks.setup_task(cfg)


def get_dataset(
    config_path: str | Path,
    task: Optional[BaseTask] = None,
    cfg: Optional[Config] = None,
) -> tuple[Config, BaseTask, dict]:
    if task is None:
        cfg, task = get_task(str(config_path), cfg)

    return cfg, task, task.build_datasets(cfg)


def get_model(
    config_path: str | Path,
    task: Optional[BaseTask] = None,
    cfg: Optional[Config] = None,
) -> tuple[Config, BaseTask, Blip2Qformer]:
    if task is None:
        cfg, task = get_task(str(config_path), cfg)

    return cfg, task, task.build_model(cfg)  # .to(torch.float16)


def get_loader(
    config_path: str | Path, shuffle: bool = False
) -> dict[str, PrefetchLoader]:
    cfg, _, _datasets = get_dataset(str(config_path))

    datasets = list(_datasets.values())[0]

    run_cfg = cfg.run_cfg
    loader: dict[str, PrefetchLoader] = {}

    for phase in ["train", "val", "test"]:
        dataset = datasets[phase]

        collate_fn = getattr(dataset, "collater")
        num_workers = run_cfg["num_workers"]
        if phase == "train":
            bs = run_cfg["batch_size_train"]
        else:
            bs = run_cfg["batch_size_eval"]

        loader[phase] = PrefetchLoader(
            DataLoader(
                dataset,
                batch_size=bs,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=True,
                sampler=None,
                collate_fn=collate_fn,
                drop_last=False,
            )
        )

    return loader


def init(
    config_path: str | Path = "blip2_flickr.yaml",
) -> tuple[Config, BaseTask, dict, torch.nn.Module, RunnerBase]:
    _config_path = str(config_path)
    job_id = now() + str(time())

    cfg = get_config(_config_path)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    setup_logger()

    _, task = get_task(_config_path, cfg)
    _, _, datasets = get_dataset(_config_path, task, cfg)
    model = task.build_model(cfg).to(torch.float16)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    return cfg, task, datasets, model, runner
