#!/usr/bin/env bash

# Where the ImageNet2012 TFR is stored to. Replace this with yours
DATA_DIR=/home1/irteam/user/jklee/food-fighters/models/tensorflow/imagenet_raw/validation_292_256

# Where the the checkpoint to evaluate is saved to. Replace this with yours
MODEL_DIR=hdfs://c3/user/tapi/sweaterr/train_imagenet/m16blskaa_mixupaa_drbl_kde_ep600

python mce/eval_robustness.py \
--robustness_type=ce \
--gpu_index=0,1,2,3,4,5,6,7 \
--num_classes=1001 \
--batch_size=256 \
--resnet_size=152 \
--image_size=256 \
--bl_alpha=1 \
--bl_beta=2 \
--resnet_version=2 \
--anti_alias_type=sconv \
--anti_alias_filter_size=3 \
--data_format=channels_first \
--use_sk_block=True \
--label_file=datasets/imagenet_lsvrc_2015_synsets.txt \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR}