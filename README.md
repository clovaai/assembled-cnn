# Compounding the Performance Improvements of Assembled Techniques in a Convolutional Neural Network


[paper](TBA) | [pretrained model](https://drive.google.com/drive/folders/1o8vj8_ZOPByjRKZzRPZMbuoKyxIwd_IZ?usp=sharing)
Official Tensorflow implementation  

> [Jungkyu Lee](mailto:jungkyu.lee@navercorp.com), [Taeryun Won](mailto:lory.tail@navercorp.com), [Kiho Hong](mailto:kiho.hong@navercorp.com)<br/>
> Clova Vision, NAVER Corp.



**Abstract**

*Recent studies in image classification have demonstrated a variety of techniques for improving the performance 
of Convolutional Neural Networks (CNNs). However, attempts to combine existing techniques to create a practical model 
are still uncommon. In this study, we carry out extensive experiments to validate that carefully assembling these techniques 
and applying them to a basic CNN model in combination can improve the accuracy and robustness of the model while minimizing 
the loss of throughput. For example, our proposed ResNet-50 shows an improvement in top-1 accuracy from 76.3% to 82.78%, 
and mCE improvement from 76.0% to 48.9%, on the ImageNet ILSVRC2012 validation set. With these improvements, inference 
throughput only decreases from 536 to 312. The resulting model significantly outperforms state-of-the-art models with similar 
accuracy in terms of mCE and inference throughput. To verify the performance improvement in transfer learning, 
fine grained classification and image retrieval tasks were tested on several open datasets and showed that the improvement 
to backbone network performance boosted transfer learning performance significantly. Our approach achieved 1st place 
in the iFood Competition Fine-Grained Visual Recognition at CVPR 2019.*

<p align="center">
  <img src="./figures/summary_architecture.png" align="center" width="500" title="summary_table" >
</p>

## Main Results

### Summary of key results

<p align="center">
 <img src="./figures/summary_table.png" align="center" width="500" title="summary_table" >
</p>

### Ablation Study

<p align="center">
  <img src="./figures/ablation_study_imagenet.png" align="center" width="1000" title="summary_table">
</p>

## Honors

Based on our repository, we achieved 1st place in [iFood Competition Fine-Grained Visual Recognition at CVPR 2019](https://www.kaggle.com/c/ifood-2019-fgvc6/leaderboard).

## Getting Started

* This work was tested with Tensorflow 1.14.0, CUDA 10.0, python 3.6.

### Requirements

```bash
pip install Pillow sklearn requests Wand tqdm
```


### Data preparation

We assume you already have the following data:
* ImageNet2012 raw images and tfrecord. For this data, please refer to [here](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
* For knowledge distillation, you need to add the teacher's logits to the TFRecord according to [here](./kd/README.md)
* For transfer learing datasets, refer to scripts in [here](./datasets)
* To download pretrained model, visit [here](https://drive.google.com/drive/folders/1o8vj8_ZOPByjRKZzRPZMbuoKyxIwd_IZ?usp=sharing)
* To make mCE evaluation dataset. refer to [here](./datasets/CE_dataset/README.MD)

### Reproduce Results

First, download pretrained models from [here](https://drive.google.com/drive/folders/1o8vj8_ZOPByjRKZzRPZMbuoKyxIwd_IZ?usp=sharing).

```bash
DATA_DIR=/path/to/imagenet2012/tfrecord
MODEL_DIR=/path/pretrained/checkpoint
CUDA_VISIBLE_DEVICES=1 python main_classification.py \
--eval_only=True \
--dataset_name=imagenet \
--data_dir=${DATA_DIR} \
--model_dir=${MODEL_DIR} \
--preprocessing_type=imagenet_224_256a \
--resnet_version=2 \
--resnet_size=152 \
--bl_alpha=1 \
--bl_beta=2 \
--use_sk_block=True \
--anti_alias_type=sconv \
--anti_alias_filter_size=3 
```

The expected final output is:

```
| accuracy:   0.841860 |
```

## Training a model from scratch.

For training parameter information, refer to [here](./nets/hparams_config.py)

Train vanila ResNet50 on ImageNet from scratch.

```console
$ ./scripts/train_vanila_from_scratch.sh
```

Train all-assemble ResNet50 on ImageNet from scratch.

```console
$ ./scripts/train_assemble_from_scratch.sh
```

## Fine-tuning the model.

In the previous section, you train the pretrained model from scratch.
You can also download pretrained model to finetune from [here](https://drive.google.com/drive/folders/1o8vj8_ZOPByjRKZzRPZMbuoKyxIwd_IZ?usp=sharing).

Fine-tune vanila ResNet50 on Food101.

```console
$ ./scripts/finetuning_vanila_on_food101.sh
```

Train all-assemble ResNet50 on Food101.

```console
$ ./scripts/finetuning_assemble_on_food101.sh
```


## mCE evaluation

You can calculate mCE on the trained model as follows: 

```console
$ ./eval_assemble_mCE_on_imagenet.sh
```

 
## Acknowledgements
This implementation is based on these repository:
* resnet official: https://github.com/tensorflow/models/tree/master/official/r1/resnet
* mce: https://github.com/hendrycks/robustness
* autoaugment: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py

## License

```
   Copyright 2020-present NAVER Corp.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
