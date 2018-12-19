from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

from syxnet.core import utils

#局部常量
_META_ARCHITECTURE_SCOPE = 'meta_architecture'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_OP = 'op'
_CONV = 'conv'
_PYRAMID_POOLING = 'pyramid_pooling'
_KERNEL = 'kernel'
_RATE = 'rate'
_GRID_SIZE = 'grid_size'
_TARGET_SIZE = 'target_size'
_INPUT = 'input'


def dense_prediction_cell_hparams():
    """DensePredictionCell 超参数

    :return:
        一个用于 dense prediction cell 的超参数字典，其key为:
        -reduction_size: Integer,单元内每个操作的输出过滤器数。
        -dropout_on_concat_features:Boolean, 是否在级联特征时使用dropout
        -dropout_on_projection_features:Boolean, 是否在投影时使用dropout
        -dropout_keep_prob:Float, 当'dropout_on_concat_features' 或 'dropout_on_projection_features'为真时，
        keep_prob的值用于dropout操作
        -concat_channels:Integer, 级联的特征将通道数减少到'concat_channels'
        -conv_rate_multiplier:Interger,用于乘以卷积率.当output_stride从16改变到8时，这个是有用的，我们需要相应地双倍卷积率
    """
    return {
        'reduction_size' : 256,
        'dropout_on_concat_features': True,
        'dropout_on_projection_features': False,
        'dropout_keep_prob': 0.9,
        'concat_channels': 256,
        'conv_rate_multiplier': 1,
    }

class DensePredictionCell(object):
    """DensePredictionCell类作为语义分割的一个层  """

    def __init__(self, config, hparams=None):
        """初始化密集预测单元
        Args:
        :param config:存储密集预测单元结构的字典
        :param hparams: 由用户提供的超参数字典。这个字典用来更新dense_prediction_cell_hparams()返回的默认字典
        """
        self.hparams = dense_prediction_cell_hparams()
        if hparams is not None:
            self.hparams.update(hparams)
        self.config = config

        #检查超参数是否有效
        if self.hparams['conv_rate_multiplier'] < 1:
            raise ValueError('conv_rate_multiplier cannot have value <1.')

    def _get_pyramid_pooling_arguments(self,
                                       crop_size,
                                       output_stride,
                                       image_grid,
                                       image_pooling_crop_size=None):
        """获取金字塔池化的参数

        :param crop_size:两个整数的列表,[crop_height, crop_width]指定整个补丁裁剪的大小
        :param output_stride: Integer,提取特征的输入步幅
        :param image_grid: 两个整数的列表.[image_grid_height, image_grid_width], 指定将如何执行金字塔池化的网格大小
        :param image_pooling_crop_size: 两个整数的列表[crop_height, crop_width]指定图像池化操作的裁剪大小.注意,将整个
        补丁的crop_size和image_pooling_crop_size分离,这样可以执行具有不同裁剪大小的image_pooling
        :return:
        一个列表(resize_value, pooled_kernel)
        """
        resize_height = utils.scale_dimension(crop_size[0], 1./output_stride)
        resize_width = utils.scale_dimension(crop_size[1], 1./output_stride)

        #如果未指定image_pooling_crop_size,使用crop_size
        if image_pooling_crop_size is None:
            image_pooling_crop_size = crop_size
        pooled_height = utils.scale_dimension(image_pooling_crop_size[1],
                                              1./(output_stride * image_grid[0]))
        pooled_width = utils.scale_dimension(image_pooling_crop_size[1],
                                             1./(output_stride * image_grid[1]))
        return ([resize_height, resize_width], [pooled_height, pooled_width])

    def _parse_operation(self, config, crop_size, output_stride,
                          image_pooling_crop_size=None):
        """解析一个操作
        当‘operation’ 是'pyramid_pooling',我们计算所需的超参数并保存在config中

        Args:
        :param config:存储一个操作所需的超参数的字典
        :param crop_size:两个整数的列表,[crop_height, crop_width]指定整个补丁的裁剪大小
        :param output_stride:Integer, 提取特征的输出步长的值
        :param image_pooling_crop_size:两个整数的列表,[crop_height, crop_width]指定用于图像池化操作的裁剪大小;
        注：将整个补丁的crop_size和image_pooling_crop_size分离,这样可以执行具有不同裁剪大小的image_pooling

        :return:一个存放相关操作信息的字典
        """
        if config[_OP] == _PYRAMID_POOLING:
            (config[_TARGET_SIZE],
             config[_KERNEL]) = self._get_pyramid_pooling_arguments(crop_size=crop_size,
                                                                    output_stride=output_stride,
                                                                    image_grid=config[_GRID_SIZE],
                                                                    image_pooling_crop_size=image_pooling_crop_size)
        return config

    def build_cell(self,
                   features,
                   output_stride=16,
                   crop_size=None,
                   image_pooling_crop_size=None,
                   weight_decay=0.00004,
                   reuse=None,
                   is_training=False,
                   fine_tune_batch_norm=False,
                   scope=None):
        """基于config建立密集预测单元

        Args:
        :param features:输入特征图,大小为[batch, height, width, channels]
        :param output_stride: Integer, 提取特征的输出步长
        :param crop_size: 一个列表[crop_height, crop_width],决定输入特征分辨率
        :param image_pooling_crop_size: 两个整数的列表,[crop_height, crop_width]指定用于图像池化操作的裁剪大小
        :param weight_decay:Float, 模型变量的权重衰减
        :param reuse:是否重复使用模型变量
        :param is_training:Boolean. 是否训练
        :param fine_tune_batch_norm:Boolean, 是否微调BN参数
        :param scope:可选数组，指定变量范围

        :return:
        特征,经过构建好的密集预测单元的特征，其形状为：[batch, height, width, channels],channels由dense_prediction_cell_hparams()
        返回的reduction_size决定
        Raises:
        ValueErroe:使用的卷积核大小不是1×1,3×3或者操作不可识别
        """
        batch_norm_params ={
                            'is_training': is_training and fine_tune_batch_norm,
                            'decay':0.9997,
                            'epsilon':1e-5,
                            'scale':True}
        hparams = self.hparams
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            padding='SAME',
                            stride=1,
                            reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope(scope, _META_ARCHITECTURE_SCOPE, [features]):
                    depth = hparams['reduction_size']
                    branch_logits =[]
                    for i, current_config in enumerate(self.config):
                        scope = 'branch%d' % i
                        current_config = self._parse_operation(config=current_config,
                                                                crop_size=crop_size,
                                                                output_stride=output_stride,
                                                                image_pooling_crop_size=image_pooling_crop_size)
                        tf.logging.info(current_config)
                        if current_config[_INPUT] < 0:
                            operation_input = features
                        else:
                            operation_input = branch_logits[current_config[_INPUT]]
                        if current_config[_OP] == _CONV:
                            if current_config[_KERNEL] == [1,1] or current_config[_KERNEL] == 1:
                                branch_logits.append(slim.conv2d(operation_input, depth, 1, scope=scope))
                            else:
                                conv_rate =[r * hparams['conv_rate_multiplier'] for r in current_config[_RATE]]
                                branch_logits.append(utils.split_separable_conv2d(operation_input,
                                                                                  filters=depth,
                                                                                  kernel_size=current_config[_KERNEL],
                                                                                  rate=conv_rate,
                                                                                  weight_decay=weight_decay,
                                                                                  scope=scope))
                        elif current_config[_OP] == _PYRAMID_POOLING:
                            pooled_features = slim.avg_pool2d(operation_input,
                                                              kernel_size=current_config[_KERNEL],
                                                              stride=[1,1],
                                                              padding='VALID')
                            pooled_features = slim.conv2d(pooled_features,
                                                          depth,
                                                          1,
                                                          scope=scope)
                            pooled_features = tf.image.resize_bilinear(pooled_features,
                                                                       current_config[_TARGET_SIZE],
                                                                       align_corners=True)
                            #如果resize_height/resize_width不是tensor的话,设置他们的形状
                            resize_height = current_config[_TARGET_SIZE][0]
                            resize_width = current_config[_TARGET_SIZE][1]
                            if isinstance(resize_height, tf.Tensor):
                                resize_height = None
                            if isinstance(resize_width, tf.Tensor):
                                resize_width = None
                            pooled_features.set_shape([None, resize_height, resize_width, depth])
                            branch_logits.append(pooled_features)
                        else:
                            raise  ValueError('Unrecognized operation.')
                    #合并分支logits
                    concat_logits = tf.concat(branch_logits, 3)
                    if self.hparams['dropout_on_concat_features']:
                        concat_logits = slim.dropout(concat_logits,
                                                     keep_prob=self.hparams['dropout_keep_prob'],
                                                     is_training=is_training,
                                                     scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
                    concat_logits = slim.conv2d(concat_logits,
                                                self.hparams['concat_channels'],
                                                1,
                                                scope=_CONCAT_PROJECTION_SCOPE)
                    if self.hparams['dropout_on_projection_features']:
                        concat_logits = slim.dropout(concat_logits,
                                                     keep_prob=self.hparams['dropout_keep_prob'],
                                                     is_training=is_training,
                                                     scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
                    return concat_logits



