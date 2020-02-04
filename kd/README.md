
# KD walkthrough

We do not compute the logit of the teacher in real time, but compute the logit of the teacher offline and make it a TFRecord.
it can't calculate logit based on the data augmentation performed in real time, but in our experiment it worked fine.



0. We assume that you prepare raw images and TFRecord files ImageNet2012 in the following path.

```bash
IMAGENET_TFR_PATH= # replace this with yours
IMAGENET_RAW_PATH= # replace this with yours
```

1. Extract logit from amoebaNet with the following command:

```
Examples: amoebanet
LOGITS_PATH=./amoebanet_logits # replace this with yours
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=0 --no_shard=0 --total_num_shard=8 & 
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=1 --no_shard=1 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=2 --no_shard=2 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=3 --no_shard=3 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=4 --no_shard=4 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=5 --no_shard=5 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=6 --no_shard=6 --total_num_shard=8 &
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=7 --no_shard=7 --total_num_shard=8 
python kd/extract_embeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH}_val --gpu_to_use=0 --no_shard=0 --total_num_shard=1 --n_tfrecord=128 --is_training=False


Examples: efficientnet
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=0 --no_shard=0 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=1 --no_shard=1 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=2 --no_shard=2 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=3 --no_shard=3 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=4 --no_shard=4 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=5 --no_shard=5 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=6 --no_shard=6 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=7 --no_shard=7 --total_num_shard=8 --net_name=efficientnet --offset=1 &
python kd/extract_embbeddings.py --data_dir=${IMAGENET_TFR_PATH} --output_dir=${LOGITS_PATH} --gpu_to_use=0 --no_shard=0 --total_num_shard=1 --n_tfrecord=128 --is_training=False --net_name=efficientnet --offset=1 &
```

2. Insert the extracted logit when creating an imagenet tfrecord.

```
IMAGENET_TFR_WITH_KD_TEACHER=imagenet_kd_efficientnet # replace/your/path
LOGITS_PATH=./amoebanet_logits # replace this with yours
python datasets/build_imagenet_data.py \
--train_directory=${IMAGENET_RAW_PATH}/train \
--validation_directory=${IMAGENET_RAW_PATH}/validation \
--make_train=True \
--output_directory=${IMAGENET_TFR_WITH_KD_TEACHER} \
--imagenet_metadata_file=datasets/imagenet_metadata.txt \
--labels_file=datasets/imagenet_lsvrc_2015_synsets.txt \
--bounding_box_file=datasets/imagenet_2012_bounding_boxes.csv \
--logits_file_path=${LOGITS_PATH}
```


3. Train the imageNet backbone with ${IMAGENET_TFR_WITH_KD_TEACHER} and give the `kd_temp` > 0 option to apply the KD loss.