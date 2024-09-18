# Object-Aware Query Perturbation for Cross-Modal Image-Text Retrieval (ECCV2024)

This is the official PyTorch impelementation of our paper ["Object-Aware Query Perturbation for Cross-Modal Image-Text Retrieval"](https://arxiv.org/abs/2407.12346) (ECCV2024).

## Prerequirements

- Python 3.10
- Poetry 1.7.1
- NVIDIA CUDA 11.8

To install Python dependencies, run `poetry install`.

## Dataset preparataion

### Flickr30K dataset

1. Download [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and unzip the file.
1. Download [Flickr30k Entitie dataset](https://github.com/BryanPlummer/flickr30k_entities).
   - There is a downlaod script in `data/fetch_entities.sh`.
1. Change `path-to-dir` to the dataset root in `examples/blip2_flickr.yaml`.

The dataset directory must be structured as follows:

```text
flickr30k/
  └── flickr30k-images
    ├── <image_id>.jpg
    ├── ...
  └── flickr30k_entities
    └── Annotations
      ├── <image_id>.xml
      ├── ...
    └── Sentences
      ├── <image_id>.txt
      ├── ...
```

### COCO dataset

1. Download val/test [MSCOCO dataset](https://cocodataset.org/#home) and unzip them.
   - There is a download script in `data/fetch_coco.sh`
1. Download [COCO Entities](https://github.com/aimagelab/show-control-and-tell) dataset.
   - There is a download script in `data/fetch_entities.sh`
1. Convert the COCO Entities dataset to the Flickr30k Entities datasets' format
   - There is a conversion script in `data/prepare_coco_entities.py`
   - Please modify the variables for the two paths highlighted as FIXME to suit your environment.
1. Change `path-to-dir` to the dataset root in `examples/blip2_coco.yaml`.

The dataset directory must be structured as follows:

```text
coco/
  └── val2014
    ├── <image_id>.jpg
    ├── ...
  └── coco_entities
    └── Annotations
      ├── <image_id>.xml
      ├── ...
    └── Sentences
      ├── <image_id>.txt
      ├── ...
```

## Run the code

### I2T/T2I evaluataion on Flickr30K dataset

\*Please modify the paths (`path-to-dir`) to the dataset directory and the configuration file in the following python script, before running the evaluation.

We can run the evaluation on Flickr30K dataset with the following script.

```shell
poetry run python examples/qpert_blip2_flickr_ret_eval.py
```

The code should output:

```markdown
# Text to Image
## R@K
R@1: 89.88, R@5: 98.14, R@10: 99.06
# Image to Text
## R@K
R@1: 97.70, R@5: 100.00, R@10: 100.00

small,
# Text to Image
## R@K
R@1: 85.33, R@5: 94.67, R@10: 94.67
# Image to Text
## R@K
R@1: 93.33, R@5: 100.00, R@10: 100.00
```

The above evaluation could take hours.
Setting `use_itm=False` speeds up the evaluation considerably, to ten to twenty minutes. (but the results will be lower).

### Feature extraction by Query-Perturbation

Please refer to the `notebooks/qpert_blip2_feat_extract.ipynb`

## Citation

```bibtex
@inproceeedings{sogi2024qpert,
  author={Sogi, Naoya and Shibata, Takashi and Terao, Makoto},
  title={{Object-Aware Query Perturbation for Cross-Modal Image-Text Retrieval}},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
}
```
