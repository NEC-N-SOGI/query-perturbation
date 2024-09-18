"""This code convert COCO-Entities dataset to Flickr-Entities format.

"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element

import lavis.tasks as tasks
import numpy as np
from lavis.common.config import Config


def parse_args(config_path: str) -> argparse.Namespace:
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

    args = parser.parse_args(args=["--cfg-path", config_path])

    return args


def prettify(elem: Element) -> str:
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty = re.sub(r"[\t ]+\n", "", reparsed.toprettyxml(indent="\t"))
    pretty = pretty.replace(">\n\n\t<", ">\n\t<")
    return pretty


def export_captions(image_ann: dict, image_name: str, root: Path) -> dict:
    nouns = []
    for k, v in image_ann.items():
        nouns.extend(v["detections"].keys())
    nouns = list(set(nouns))
    noun2idx = {noun: i for i, noun in enumerate(nouns)}

    # replace nouns in caption with [[/EN#idx/hoge noun]
    captions = []
    for (
        caption,
        v,
    ) in image_ann.items():  # list(image_ann.keys())[0], list(image_ann.values())[0]
        nouns = list(v["detections"].keys())
        caption = re.sub("\[.*?\]", "", caption)
        for i, noun in enumerate(nouns):
            caption = re.sub(
                rf"\b{noun}s?\b", f"[/EN#{noun2idx[noun]}/hoge {noun}]", caption
            )

        captions.append(caption + " .\n")

    with open(
        root / f"COCO_val2014_{int(image_name):012d}.txt",
        "w",
    ) as f:
        f.writelines(captions)

    return noun2idx


def export_annotations(
    image_name: str, img_shapes: dict, image_ann: dict, noun2idx: dict, root: Path
) -> None:
    xml = ET.Element("annotation")
    filename = ET.SubElement(xml, "filename")
    filename.text = f"COCO_val2014_{int(image_name):012d}.jpg"

    # image_shape
    size = ET.SubElement(xml, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_shapes[image_name][0])

    height = ET.SubElement(size, "height")
    height.text = str(img_shapes[image_name][1])

    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    pushed_boxs = None
    for (
        caption,
        v,
    ) in image_ann.items():
        for det_class, boxs in v["detections"].items():
            for box in boxs:
                try:
                    sim = np.abs(pushed_boxs - np.asarray(box[1])).sum(1).min()
                    if sim < 0.1:
                        continue
                except:
                    pass
                _box = box[1]
                obj = ET.SubElement(xml, "object")
                name = ET.SubElement(obj, "name")
                name.text = str(noun2idx[det_class])

                bndbox = ET.SubElement(obj, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(_box[0])
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(_box[1])
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(_box[2])
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(_box[3])

                if pushed_boxs is None:
                    pushed_boxs = np.asarray(_box)
                else:
                    pushed_boxs = np.vstack((pushed_boxs, _box))
    # save xml
    with open(
        root / f"COCO_val2014_{int(image_name):012d}.xml",
        "w",
    ) as f:
        f.write(prettify(xml))


if __name__ == "__main__":
    lavis_coco_cfg = "path-to-directory/blip2_coco.yaml"  # FIXME:
    root = Path("./coco/")  # FIXME:
    sentence_root = root / "coco_entities/Sentences"
    annotation_root = root / "coco_entities/Annotations"

    cfg = Config(parse_args(lavis_coco_cfg))
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    with open(root / "coco_entities_release.json") as f:
        data = json.load(f)

    with open(root / "datasets/coco/coco_img_shapes.json", "r") as f:
        img_shapes = json.load(f)

    os.makedirs(sentence_root, exist_ok=True)
    os.makedirs(annotation_root, exist_ok=True)

    for image_name, image_ann in data.items():
        # export captions
        noun2idx = export_captions(image_ann, image_name, sentence_root)

        # export bboxes
        # ---
        export_annotations(
            image_name, img_shapes, image_ann, noun2idx, annotation_root
        )

    missing_data = ["413120", "201732", "463918", "132317", "35498"]

    for name in missing_data:
        # find captions from datasets
        for i in datasets["coco_retrieval"]["test"].annotation:
            if f"COCO_val2014_{int(name):012d}.jpg" in i["image"]:
                captions = i["caption"]
                captions = [s + " \n" for s in captions]
                image_id = i["instance_id"]
                break

        with open(
            root / f"coco_entities/Sentences/COCO_val2014_{int(name):012d}.txt", "w"
        ) as f:
            f.writelines(captions)

        export_annotations(name, img_shapes, {}, [], annotation_root)
