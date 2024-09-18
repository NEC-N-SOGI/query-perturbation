#!/usr/bin/env bash

# COCO entities
cd coco
wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/dataset_coco.tgz
tar -xzvf dataset_coco.tgz
wget https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/coco_entities_release.json

cd ..
# Flickr 30k entities
cd flickr30k
git clone https://github.com/BryanPlummer/flickr30k_entities.git

cd flickr30k_entities
unzip annotations.zip
