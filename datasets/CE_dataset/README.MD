# Generate mCE dataset

We assume that you prepare raw images ImageNet2012 in the following path.

```bash
# replace this with yours
IMAGENET_RAW_PATH=/home1/irteam/user/jklee/food-fighters/models/tensorflow/imagenet_raw 
```

You can create an mCE evaluation set for ImageNet using the following command:

```bash
python CE_validation_generator.py \
--validation_dir=${IMAGENET_RAW_PATH}/validation \
--output_dir=${IMAGENET_RAW_PATH}/validation_256_224 \
--RESIZE_SIZE=256 \
--CROP_SIZE=224 
```
`--RESIZE_SIZE=292` and `--CROP_SIZE=256` means that 
first  resize  the  shorter  size  of each  image  to  256  pixels  while  the  aspect  ratio  is  main-tained, then  we center crop image of 224x224 size.


When evaluating model with BigLittleNet, 
since the resolution is 256x256 for validation, 
make validation set using the following command:

```bash
python CE_validation_generator.py \
--validation_dir=${IMAGENET_RAW_PATH}/validation \
--output_dir=${IMAGENET_RAW_PATH}/validation_292_256 \
--RESIZE_SIZE=292 \
--CROP_SIZE=256 
```
