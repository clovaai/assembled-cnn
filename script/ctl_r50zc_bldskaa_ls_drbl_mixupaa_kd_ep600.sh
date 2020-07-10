Total params: 41,925,801
Trainable params: 41,848,489
Non-trainable params: 77,312


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 resnet_ctl_imagenet_main.py \
--data_dir=imagenet_kd_amoebanet \
--model_dir=hdfs://c3/user/sweaterr/train_imagenet5/v1.1.5_r50zc_bldskaa_ls_drbl_mixupaa_kd_ep600_1 \
--zero_gamma \
--learning_rate_decay_type=cosine \
--use_bl \
--use_resnet_d \
--use_sk_block \
--anti_alias_type='sconv' \
--label_smoothing=0.1 \
--use_dropblock \
--mixup_alpha=0.2 \
--autoaugment_type=imagenet \
--kd_temp=1.0 \
--train_epochs=600 \
--dtype=fp16 \
--num_gpus=8 \
--batch_size=1024 \
--base_learning_rate=0.4 \
--distribution_strategy=mirrored \
--weight_decay=0.00005 \
--bn_momentum=0.997 \
--lr_warmup_epochs=5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 resnet_ctl_imagenet_main.py \
--data_dir=imagenet_kd_amoebanet \
--model_dir=hdfs://c3/user/sweaterr/train_imagenet5/v1.1.5.1_r50zc_bldskaa_ls_drbl_mixupaa_kd_ep600 \
--zero_gamma \
--learning_rate_decay_type=cosine \
--use_bl \
--use_resnet_d \
--use_sk_block \
--anti_alias_type='sconv' \
--label_smoothing=0.1 \
--use_dropblock \
--mixup_alpha=0.2 \
--autoaugment_type=imagenet \
--kd_temp=1.0 \
--train_epochs=600 \
--dtype=fp16 \
--num_gpus=8 \
--batch_size=1024 \
--base_learning_rate=0.4 \
--distribution_strategy=mirrored \
--weight_decay=0.00005 \
--bn_momentum=0.997 \
--lr_warmup_epochs=5