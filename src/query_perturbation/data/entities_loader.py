import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def read_sentence_file(path: str) -> dict[str, list[str]]:
    sentence_path = path.replace("Annotations", "Sentences").replace(".xml", ".txt")
    with open(sentence_path, "r") as f:
        sentences = f.readlines()

    # extract words in [] and remove redundant strings.
    def extract_word(s: str) -> str:
        return " ".join(s.split(" ")[1:])[:-1]

    def extract_id(s: str) -> str:
        return s.split("#")[1].split("/")[0]

    words: dict[str, Any] = {}
    for sentence in sentences:
        _words = re.findall(r"\[.*?\]", sentence)
        items = {extract_id(i): extract_word(i) for i in _words}

        for k, v in items.items():
            if words.get(k) is None:
                words[k] = set([v])
            else:
                words[k] |= set([v])

    words_list: dict[str, list[str]] = {}
    for k, v in words.items():
        words_list[k] = list(v)
    return words_list


def get_coord(container: ET.Element, names: list[str]) -> list[float]:
    return [float(str(container.find(name).text)) - 1 for name in names]  # type: ignore


def get_img_size(root: ET.Element) -> tuple[int, int]:
    for _size_info in root.find("size"):  # type: ignore
        tag = _size_info.tag
        if tag == "width":
            width = int(str(_size_info.text))
        elif tag == "height":
            height = int(str(_size_info.text))

    return height, width


def read_annotation_file(
    xml_path: str | Path,
) -> tuple[int, int, dict[str, list[list[float]]]]:
    """

    Args:
        xml_path (str | Path): path to an annotation file

    Returns:
        tuple[int, int, dict[str, list[list[float]]]]: [img height, img widht, bboxes]. each bbox is [xmin, ymin, xmax, ymax]
    """
    root = ET.parse(xml_path).getroot()
    height, width = get_img_size(root)

    boxes: dict[str, list[list[float]]] = {}
    for objs in root.findall("object"):
        bndbox = objs.findall("bndbox")

        # no bbox
        if len(bndbox) == 0:
            continue

        # obj name
        box_name = ",".join([str(names.text) for names in objs.findall("name")])
        # NEW obj name
        if box_name not in boxes:
            boxes[box_name] = []

        # Bbox info
        boxes[box_name].append(
            get_coord(bndbox[0], ["xmin", "ymin", "xmax", "ymax"])
        )

    return height, width, boxes
