#!/usr/bin/env bash

# Where the ImageNet2012 validation for mCE is stored to. Replace this with yours
DATA_DIR=/home1/irteam/user/jklee/food-fighters/models/tensorflow/imagenet_raw/validation_256_224

MODEL_DIR=hdfs://c3/user/tapi/sweaterr/train_imagenet/m2_fp16

python mce/eval_robustness.py \
--robustness_type=ce \
--gpu_index=4,5,6,7 \
--num_classes=1001 \
--batch_size=1024 \
--image_size=224 \
--resnet_size=50 \
--resnet_version=1 \
--data_format=channels_first \
--label_file=datasets/imagenet_lsvrc_2015_synsets.txt \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR}