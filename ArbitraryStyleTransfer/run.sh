#!/bin/bash

#### training command example
#CUDA_VISIBLE_DEVICES=0 python train.py --content_dir /path_to/coco/train2017 --style_dir /path_to/WikiArt/train --save_dir ./experiments_efdm --log_dir ./log_efdm  --style efdm

## test command example
#CUDA_VISIBLE_DEVICES=0 python test.py --content_dir input/content --style_dir input/style --style_type efdm --test_style_type efdm --crop --output output --decoder ./experiments_efdm/decoder_iter_160000.pth.tar
#CUDA_VISIBLE_DEVICES=0 python test.py --content_dir input2/content --style_dir input2/style --style_type efdm --test_style_type efdm --crop --output photo --decoder ./experiments_efdm/decoder_iter_160000.pth.tar
