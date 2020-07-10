# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.utils.flags import core as flags_core


def define_metric_learning_flags(flags):
  define_common_flags(flags)
  flags.DEFINE_string(
    name='metric_loss_type', default='npair',
    help=flags_core.help_wrap('[npair|angular|npair_and_angular]'))
  flags.DEFINE_integer(
    name='dim_features', short_name='dimf', default=64,
    help=flags_core.help_wrap('Dimension of features.'))
  flags.DEFINE_boolean(
    name='return_embedding', default=False,
    help=flags_core.help_wrap('마지막 fc를 사용하지 않는다.'))
  flags.DEFINE_boolean(
    name='restore_embedding', default=False,
    help=flags_core.help_wrap('`restore_embedding`를 True로 하면 분류에서 학습한 embedding의 weight를  warm-start 한다.'))
  flags.DEFINE_integer(
    name='angular_degree', default=45,
    help=flags_core.help_wrap('A degree for angular loss.'))

def define_common_flags(flags):
  flags.DEFINE_string(
    name='dataset_name', default=None,
    help=flags_core.help_wrap('imagenet, food100, food101, naver_food547, cub_200_2011'))

  ################################################################################
  # Basic Hyperparameters
  ################################################################################
  flags.DEFINE_integer(
    name='val_batch_size', default=256,
    help=flags_core.help_wrap(
      'The number of steps to warmup learning rate.'))
  flags.DEFINE_string(
    name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
    help=flags_core.help_wrap(
      'If not None initialize all the network except the final layer with '
      'these values'))
  flags.DEFINE_integer(
    name='fine_tune', short_name='ft', default=0,
    help=flags_core.help_wrap(
      'If True do not train any parameters except for the final layer.'))
  flags.DEFINE_float(
    name='learning_rate_decay_factor', short_name='lrdf', default=0.94,
    help=flags_core.help_wrap(
      'Number of epochs after which learning rate decays.'))
  flags.DEFINE_float(
    name='base_learning_rate', short_name='blr', default=0.01,
    help=flags_core.help_wrap(
      'Base learning rate.'))
  flags.DEFINE_float(
    name='end_learning_rate', short_name='elr', default=0.0001,
    help=flags_core.help_wrap(
      'The minimal end learning rate used by a polynomial decay learning rate.'))
  flags.DEFINE_string(
    name='learning_rate_decay_type', short_name='lrdt', default='piecewise',
    help=flags_core.help_wrap(
      'Specifies how the learning rate is decayed. One of '
      '"fixed", "exponential", "polynomial", "piecewise", "cosine"'))
  flags.DEFINE_float(
    name='momentum', short_name='mmt', default=0.9,
    help=flags_core.help_wrap(
      'The momentum for the MomentumOptimizer.'))
  flags.DEFINE_float(
    name='bn_momentum', default=0.997,
    help=flags_core.help_wrap(
      'batch normalization momentum.'))
  flags.DEFINE_integer(
    name='embedding_size', default=0,
    help=flags_core.help_wrap('embedding_size > 0 시, pool 과 클래스 fc 사이에 embedding을 추가한다.'))
  flags.DEFINE_list(
    name='piecewise_lr_boundary_epochs', short_name='pwlrb', default=[30, 60, 90, 120],
    help=flags_core.help_wrap(
      'A list of ints with strictly increasing entries to reduce the learning rate at certain epochs. '
      'It\'s for piecewise lr decay type purpose only.'))
  flags.DEFINE_list(
    name='piecewise_lr_decay_rates', short_name='pwlrd',
    default=[1, 0.1, 0.01, 0.001, 1e-4],
    help=flags_core.help_wrap(
      'A list of floats that specifies the decay rates for the intervals defined by piecewise_lr_boundary_epochs. '
      'It should have one more element than piecewise_lr_boundary_epochs. It\'s for piecewise lr decay type purpose only.'))

  ################################################################################
  # Loss related
  ################################################################################
  flags.DEFINE_string(
    name='cls_loss_type', default='softmax',
    help=flags_core.help_wrap(
      '`softmax`, `sigmoid`, `focal`'))

  flags.DEFINE_string(
    name='logit_type', default=None,
    help=flags_core.help_wrap('Logit type `arc_margin` or `None`.'))

  flags.DEFINE_string(
    name='loss_weight_type', default='uniform',
    help=flags_core.help_wrap('Loss weight type, `cb` or `uniform`'))

  flags.DEFINE_float(
    name='cb_beta', default=0.999,
    help=flags_core.help_wrap(
      'Hyperparameter beta when using class-balanced loss.'))

  flags.DEFINE_string(
    name='imgcnt_dir', default='imgcnt',
    help=flags_core.help_wrap('Directory that stores the number of images by class'))

  flags.DEFINE_float(
    name='focal_loss_gamma', default=0.5,
    help=flags_core.help_wrap(
      'Hyperparameter gamma when using focal loss.'))

  flags.DEFINE_float(
    name='anchor_loss_gamma', default=0.5,
    help=flags_core.help_wrap(
      'Hyperparameter gamma when using anchor loss.'))

  flags.DEFINE_integer(
    name='anchor_loss_warmup_epochs', default=0,
    help=flags_core.help_wrap(
      'epochs of initial softmax CE training for anchor loss.(5 recommended)'))

  flags.DEFINE_float(
    name='arc_s', default=80.0,
    help=flags_core.help_wrap(
      's of arc-margin loss'))

  flags.DEFINE_float(
    name='arc_m', default=0.15,
    help=flags_core.help_wrap(
      'margin m of arc-margin loss'))

  flags.DEFINE_string(
    name='pool_type', default='gap',
    help=flags_core.help_wrap('`gap` or `gem` or `flatten`'))

  flags.DEFINE_boolean(
    name='no_downsample', default=False,
    help=flags_core.help_wrap('If true, remove downsample in group 4'))

  ################################################################################
  # Data augmentation
  ################################################################################
  flags.DEFINE_string(
    name='preprocessing_type', default='imagenet',
    help=flags_core.help_wrap(' "imagenet" or "cub"'))

  flags.DEFINE_string(
    name='autoaugment_type', default=None,
    help=flags_core.help_wrap(
      'Specifies auto augmentation type. One of "imagenet", "svhn", "cifar", "good"'
      'To use numpy implementation, prefix "np_" to the type.'))

  flags.DEFINE_float(
    name='mixup_alpha', default=0.0,
    help=flags_core.help_wrap(''))

  flags.DEFINE_integer(
    name='repeated_augment_size', default=1,
    help=flags_core.help_wrap('batch 내에서 하나의 인스턴스에 대해서 augmentation할 갯수.'))
  flags.DEFINE_boolean(
    name='use_color_jitter', default=None,
    help=flags_core.help_wrap('Whether to use color jitter'))

  flags.DEFINE_string(
    name='uda_data_dir', default=None,
    help=flags_core.help_wrap(
      ''))
  flags.DEFINE_integer(
    name='unsup_ratio', default=0,
    help=flags_core.help_wrap('The ratio between batch size of unlabeled data and labeled data, '
                              'i.e., unsup_ratio * train_batch_size is the batch_size for unlabeled data.'
                              'Do not use the unsupervised objective if set to 0.'))
  flags.DEFINE_float(
    name='uda_softmax_temp', default=-1,
    help=flags_core.help_wrap('The temperature of the Softmax when making prediction on unlabeled'
                              'examples. -1 means to use normal Softmax'))
  flags.DEFINE_float(
    name='uda_confidence_thresh', default=-1,
    help=flags_core.help_wrap('The threshold on predicted probability on unsupervised data. If set,'
                              'UDA loss will only be calculated on unlabeled examples whose largest'
                              'probability is larger than the threshold'))

  flags.DEFINE_float(
    name='ent_min_coeff', default=0,
    help=flags_core.help_wrap('0보다 크면, logit에 entropy가 커지지 않게(=confidence 가 균일해지지 않게) loss 추가'))

  flags.DEFINE_float(
    name='unsup_coeff', default=1,
    help=flags_core.help_wrap('The coefficient on the UDA loss. '
                              'setting unsup_coeff to 1 works for most settings. '
                              'When you have extermely few samples, consider increasing unsup_coeff'))
  flags.DEFINE_enum(
    "tsa", "",
    enum_values=["", "linear_schedule", "log_schedule", "exp_schedule"],
    help="anneal schedule of training signal annealing. "
         "tsa='' means not using TSA. See the paper for other schedules.")

  flags.DEFINE_string(
    name='uda_autoaugment_type', default=None,
    help=flags_core.help_wrap(
      'Specifies auto augmentation type for Unsupervised Data Augmentation.'
      'One of "imagenet", "svhn", "cifar", "good"'))
  ################################################################################
  # Network Tweak
  ################################################################################
  flags.DEFINE_integer(
    name='resnet_size', default=50,
    help=flags_core.help_wrap('The number of resnet layers: 50, 100, 150, 200'))

  flags.DEFINE_boolean(
    name='use_resnet_d', default=False,
    help=flags_core.help_wrap('Use resnet_d architecture. '
                              'For more details, refer to https://arxiv.org/abs/1812.01187'))
  flags.DEFINE_boolean(
    name='use_space2depth', default=False,
    help=flags_core.help_wrap(''))
  flags.DEFINE_boolean(
    name='use_se_block', default=False,
    help=flags_core.help_wrap('Use SE block. '
                              'For more details, refer to https://arxiv.org/abs/1709.01507'))
  flags.DEFINE_boolean(
    name='use_dedicated_se', default=False,
    help=flags_core.help_wrap('Use SE block.'))
  flags.DEFINE_boolean(
    name='use_sk_block', default=False,
    help=flags_core.help_wrap('Use SK block. '))

  flags.DEFINE_boolean(
    name='use_bl', default=False,
    help=flags_core.help_wrap('Use Big Little Net.'))

  flags.DEFINE_boolean(
    name='use_bs', default=False,
    help=flags_core.help_wrap('Blocks Selection'))

  flags.DEFINE_integer(
    name='anti_alias_filter_size', default=3,
    help=flags_core.help_wrap('Anti-aliasing filter size, One of 2, 3, 4, 5, 6, 7'))

  flags.DEFINE_string(
    name='anti_alias_type', default="",
    help=flags_core.help_wrap(
      'Specifies auto anti alias type. For example,  "max,proj,sconv" is fully anti-alias, '
      '"sconv" means that only strided conv is applied. '))

  flags.DEFINE_enum(
    name='resnet_version', short_name='rv', default='1',
    enum_values=['1', '2', '3'],
    help=flags_core.help_wrap(
      'Version of ResNet. (1 or 2 or 3) See README.md for details.'
      '3은 BigLittleNet 이다.'))

  flags.DEFINE_integer(
    name='bl_alpha', default=2,
    help=flags_core.help_wrap(''))
  flags.DEFINE_integer(
    name='bl_beta', default=4,
    help=flags_core.help_wrap(''))

  ################################################################################
  # Regularization
  ################################################################################
  flags.DEFINE_float(
    name='weight_decay', short_name='wd', default=0.0001,
    help=flags_core.help_wrap(
      'The weight decay on the model weights.'))
  flags.DEFINE_boolean(
    name='use_dropblock', default=False,
    help=flags_core.help_wrap('Use dropblock. '
                              'For more details, refer to https://arxiv.org/abs/1810.12890'))
  flags.DEFINE_list(
    name='dropblock_kp', short_name='drblkp',
    default=[1.0, 0.9],
    help=flags_core.help_wrap(
      'Initial keep_prob and end keep_prob of dropblock.'))
  flags.DEFINE_float(
    name='label_smoothing', short_name='lblsm', default=0.0,
    help=flags_core.help_wrap(
      'If greater than 0 then smooth the labels.'))
  flags.DEFINE_float(
    name='kd_temp', default=0,
    help=flags_core.help_wrap('Use knowledge distillation.'))
  flags.DEFINE_integer(
    name='num_classes', default=1001,
    help=flags_core.help_wrap(''))

  ################################################################################
  # Tricks to Learn the Models
  ################################################################################
  flags.DEFINE_integer(
    name='lr_warmup_epochs', default=0,
    help=flags_core.help_wrap('The number of learning rate warmup epochs. If 0 do not use warmup'))
  flags.DEFINE_boolean(
    name='zero_gamma', default=False,
    help=flags_core.help_wrap(
      'If True, we initialize gamma = 0 for all BN layers that sit at the end of a residual block'))
  flags.DEFINE_string(
    name='rollback_keep_checkpoint_path', default=None,
    help=flags_core.help_wrap('rollback 하지 않을 영역의 weight가 저장되어 있는 checkpoint 경로'))
  flags.DEFINE_float(
    name='rollback_lr_multiplier', default=10.0,
    help=flags_core.help_wrap('rollback 사용시, rollback하지 않는 영역은 learning_rate에 rollback_lr_multiplier을 나눈다.'))
  flags.DEFINE_integer(
    name='rollback_period', default=None,
    help=flags_core.help_wrap('1 이상이면, Rolling back fine-tuning 기능을 사용한다.'
                              '자세한 사항은 https://arxiv.org/pdf/1901.06140.pdf'))

  ################################################################################
  # Others
  ################################################################################
  flags.DEFINE_string(
    name='dct_method', short_name='dctm', default='INTEGER_ACCURATE',
    help=flags_core.help_wrap(
      'An optional `string`. '
      'Defaults to `""`. '
      'string specifying a hint about the algorithm used for decompression.  '
      'Defaults to "" which maps to a system-specific default.  '
      'Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].  '
      'The hint may be ignored (e.g., the internal jpeg library changes to a version '
      'that does not have that specific option.)'))
  flags.DEFINE_float(
    name='use_ranking_loss', default=None, help=flags_core.help_wrap(
      'if use_ranking_loss  > 0 use softmax + ranking loss'))
  flags.DEFINE_boolean(
    name='with_drawing_bbox', default=False,
    help=flags_core.help_wrap('If True, display raw images with bounding box in tensorboard'))
  flags.DEFINE_boolean(
    name='use_hyperdash', default=True,
    help=flags_core.help_wrap('hyperdash(https://hyperdash.io/) 모니터링 툴을 사용한다.'))
  flags.DEFINE_integer(
    name='num_best_ckpt_to_keep', short_name='nbck', default=3,
    help=flags_core.help_wrap(
      'The number of best performance checkpoint to keep.'))
  flags.DEFINE_boolean(
    name='keep_ckpt_every_eval', default=True,
    help=flags_core.help_wrap('If True, checkpoints are saved for each evaluation.'))
  flags.DEFINE_integer(
    name='keep_checkpoint_max', default=20,
    help=flags_core.help_wrap('keep checkpoint max.'))
  flags.DEFINE_boolean(
    name='eval_only', default=False,
    help=flags_core.help_wrap('Skip training and only perform evaluation on '
                              'the latest checkpoint.'))
  flags.DEFINE_integer(
    name='eval_image_size', default=224,
    help=flags_core.help_wrap('Image size to use during inference time.'))
  flags.DEFINE_integer(
    name='eval_crop_type', default=0,
    help=flags_core.help_wrap('0 is center crop, 1 is to cut longer axis'))
  flags.DEFINE_boolean(
    name='training_random_crop', default=True,
    help=flags_core.help_wrap('Whether to randomly crop training images'))
  flags.DEFINE_boolean(
    name='export_only', default=False,
    help=flags_core.help_wrap('Skip training and evaluations.'
                              'Only export the latest checkpoint.'))
  flags.DEFINE_string(
    name='export_decoder_type', default='jpeg',
    help=flags_core.help_wrap('Specify the image decoder for binary input pb (jpeg | webp)'))
  flags.DEFINE_integer(
    name='xla_type', default=0,
    help=flags_core.help_wrap('xla_type > 0이면, CUDA9.2 이상에서 학습속도 향상'))
  flags.DEFINE_float(
    name='save_checkpoints_epochs', default=1.0,
    help=flags_core.help_wrap('Save checkpoint every save_checkpoints_epochs'))
  flags.DEFINE_float(
    name='ratio_fine_eval', default=1.0,
    help=flags_core.help_wrap('`train_epochs`*`ratio_fine_eval`에서부터는 무조건 1 epoch마다 평가한다.'))
  flags.DEFINE_boolean(
    name='zeroshot_eval', default=False,
    help=flags_core.help_wrap('zeroshot 평가'))
  
  flags.DEFINE_string(
    name='eval_similarity', default='cosine',
    help=flags_core.help_wrap('cosine or euclidean, only for lfw'))
  
  flags.DEFINE_boolean(
    name='use_test_flip', default=False,
    help=flags_core.help_wrap('test time flip image feature concatenation, only for lfw'))
  
  flags.DEFINE_boolean(
    name='tfrecords_load', default=True,
    help=flags_core.help_wrap('Wheter to use tfrecords'))
  
  flags.DEFINE_string(
    name='label_file', default=None,
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_string(
    name='train_regex', default='train-*',
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_string(
    name='val_regex', default='validation-*',
    help=flags_core.help_wrap('use label ids which has been generated by building tfrecord. '
                              'If None, it put the label ids in alphabetical order.'))
  flags.DEFINE_list(
    name='recall_at_k',
    default=[1, 5],
    help=flags_core.help_wrap(
      'A list of int that specifies the k values of the recall_at_k'))

  flags.DEFINE_string(name='last_pool_channel_type', default="gap",
                      help=flags_core.help_wrap(''))

  flags.DEFINE_integer(
    name='verbose', default=2,
    help=flags_core.help_wrap('0: slient'
                              '1: progress bar'
                              '2: one line per epoch'))

  flags.DEFINE_boolean(
    name='debug', default=False,
    help=flags_core.help_wrap('Debug model'))

