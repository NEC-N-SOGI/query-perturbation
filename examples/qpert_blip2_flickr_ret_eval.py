import argparse
from pathlib import Path

from query_perturbation.data.box_loader import ImageBoxLoader
from query_perturbation.models.blip2.blip2_qpert import BLIP2Extractor
from query_perturbation.models.weight import AreaBasedWeight, AreaWeightType
from query_perturbation.task.blip2_task import BLIP2Task

ROOT_DIR = Path("path-to-dir/flickr30k")  # FIXME:
CONF_ROOT_DIR = Path("path-to-dir/query-perturbation/examples/")  # FIXME:


parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=8)
parser.add_argument("--npc", type=float, default=0.99)

args = parser.parse_args()
npc = args.npc if args.npc < 1 else int(args.npc)


class FlickrDataCfg:
    dataset_root: Path = ROOT_DIR
    config_path: Path = CONF_ROOT_DIR / "blip2_flickr.yaml"
    entities_root: Path = ROOT_DIR / "flickr30k_entities"


cfg = FlickrDataCfg()

loader = ImageBoxLoader(cfg, "test")

weight = AreaBasedWeight(6, args.alpha, AreaWeightType.CONSTANT)
extractor = BLIP2Extractor(cfg.config_path, npc, "cuda", weight)

task = BLIP2Task(loader, extractor, use_itm=True)

overall_metrics, size_metrics = task.run()
task.pretty_print(overall_metrics)
print("small,")
task.pretty_print(size_metrics[0])
