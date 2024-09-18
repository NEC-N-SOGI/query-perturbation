import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from query_perturbation.data.dataset_cfg import BaseDataCfg
from query_perturbation.data.entities_loader import (
    read_annotation_file,
    read_sentence_file,
)
from query_perturbation.lavis_utils import get_loader


@dataclass
class BoxInfo:
    areas: list[float]  # areas of objects. normalized to [0, 1]
    boxes: list[list[float]]  # bounding boxes. boxes[i] = [xmin, ymin, xmax, ymax]
    img_size: list[int]  # original image size, [height, width]
    sentences: list[str]  # sentences of objects


class ImageBoxLoader:
    def __init__(
        self,
        data_cfg: BaseDataCfg,
        data_split: str,
        dtype: torch.dtype = torch.float32,
        shuffle: bool = False,
    ) -> None:
        loader = get_loader(data_cfg.config_path, shuffle)[data_split]
        self.loader = loader
        self.data_cfg = data_cfg
        self.entities_root = data_cfg.entities_root
        self.image_names = loader.loader.dataset.image
        self.dtype = dtype
        self.dataset = loader.dataset

    def get_box_info(
        self,
        xml_path: Path,
    ) -> BoxInfo:
        """Load entities data and convert it to BoxInfo

        Args:
            xml_path (Path): path to xml file

        Returns:
            BoxInfo: areas, bboxes, img_size, sentences
        """
        # load entities data
        try:
            height, width, boxes = read_annotation_file(str(xml_path))
            sentences = read_sentence_file(str(xml_path))
        except FileNotFoundError:
            return BoxInfo(
                areas=[-1.0],
                boxes=[],
                img_size=[0, 0],
                sentences=[""],
            )

        # remove redundant? bboxs
        boxes = {
            k: v
            for k, v in boxes.items()
            if any(list(map(sentences.get, k.split(","))))
        }

        image_area: int = height * width
        bbox = [j for i in list(boxes.values()) for j in i]
        bbox_area = list(
            map(lambda x: (x[2] - x[0]) * (x[3] - x[1]) / image_area, bbox)
        )
        if len(bbox_area) == 0:
            bbox_area = [0]

        # get sentenfces
        sentence_ids = list(set([j for i in boxes.keys() for j in i.split(",")]))
        sentences = [sentences[i] for i in sentence_ids]  # type: ignore

        return BoxInfo(
            areas=bbox_area,
            boxes=bbox,
            img_size=[height, width],
            sentences=sentences,  # type: ignore
        )

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, list[int], list[BoxInfo]]]:
        first = True
        images: torch.Tensor = torch.empty(0)

        indexs: list[int] = []
        box_infos: list[BoxInfo] = []

        for samples in self.loader:
            next_indexs: list[int] = samples["index"]
            next_images: torch.Tensor = samples["image"].to(self.dtype)

            next_box_infos = []
            for idx in next_indexs:
                image_name = self.image_names[idx]
                basename = os.path.basename(image_name).split(".")[0]
                path = self.entities_root / f"Annotations/{basename}.xml"

                next_box_infos.append(self.get_box_info(path))

            if not first:
                yield images, indexs, box_infos
            else:
                first = False

            images = next_images
            indexs = next_indexs
            box_infos = next_box_infos

        yield images, indexs, box_infos

    @property
    def __len__(self) -> int:
        return len(self.loader)
