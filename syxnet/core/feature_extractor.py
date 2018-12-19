"""为不同模型提取特征"""
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim

from syxnet.core import resnet_v1_beta
from syxnet.core import xception
from tensorflow.contrib.slim.nets import resnet_utils
from nets.mobilenet import mobilenet_v2

#用于mobileNetV2的默认结束点
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'


def _mobilenet_v2(net,
                  depth_multiplier,
                  output_stride,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
    """对mobilenet_v2的复用添加支持的辅助函数

    Args:
    :param net:输入tensor,[batch_size, height, width, channels].
    :param depth_multiplier: 所有卷积运算的深度（通道数）的浮点乘数。 该值必须大于零。
    典型用法是将此值设置为（0,1）以减少模型的参数数量或计算成本。
    :param output_stride:一个整数，指定请求的输入与输出空间分辨率之比。 如果不是None，
    那么我们在必要时调用atrous卷积以防止网络降低激活映射的空间分辨率。
    允许值为8（精确完全卷积模式），16（快速完全卷积模式），32（分类模式）。
    :param reuse:重用模型变量
    :param scope:可选变量名
    :param final_endpoint:构建网络的结束点
    :return:从MobileNetV2提取特征
    """
    with tf.variable_scope(scope,
                           'MobilenetV2', [net], reuse=reuse) as scope:
        return mobilenet_v2.mobilenet_base(net,
                                           conv_defs=mobilenet_v2.V2_DEF,
                                           depth_multiplier=depth_multiplier,
                                           min_depth=8 if depth_multiplier == 1.0 else 1,
                                           divisible_by=8 if depth_multiplier == 1.0 else 1,
                                           final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT,
                                           output_stride=output_stride,
                                           scope=scope)


#网络名到网络函数的映射
networks_map = {
    'mobilenet_v2': _mobilenet_v2,
    'resnet_v1_50': resnet_v1_beta.resnet_v1_50,
    'resnet_v1_50_beta': resnet_v1_beta.resnet_v1_50_beta,
    'resnet_v1_101': resnet_v1_beta.resnet_v1_101,
    'resnet_v1_101_beta': resnet_v1_beta.resnet_v1_101_beta,
    'xception_41':xception.xception_41,
    'xception_65':xception.xception_65,
    'xception_71':xception.xception_71,
}

#网络名和网络参数的映射
arg_scopes_map = {
    'mobilenet_v2': mobilenet_v2.training_scope,
    'resnet_v1_50': resnet_utils.resnet_arg_scope,
    'resnet_v1_50_beta':resnet_utils.resnet_arg_scope,
    'resnet_v1_101': resnet_utils.resnet_arg_scope,
    'resnet_v1_101_beta': resnet_utils.resnet_arg_scope,
    'xception_41': xception.xception_arg_scope,
    'xception_65': xception.xception_arg_scope,
    'xception_71': xception.xception_arg_scope,
}

#结束点特征的名字
DECODER_END_POINTS = 'decoder_end_points'

#一个网络名到结束点特征的字典
networks_to_feature_maps = {
    'mobilenet_v2': {
        DECODER_END_POINTS: ['layer_4/depthwise_output'],
    },
    'resnet_v1_50': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_50_beta': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_101': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_101_beta': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'xception_41':{
        DECODER_END_POINTS: ['entry_flow/block2/unit_1/xception_module/'
                             'separable_conv2_pointwise',]
    },
    'xception_65':{
        DECODER_END_POINTS: ['entry_flow/block2/unit_1/xception_module/'
                             'separable_conv2_pointwise',],
    },
    'xception_71': {
        DECODER_END_POINTS: ['entry_flow/block2/unit_1/xception_module/'
                             'separable_conv2_pointwise',],
    },
}

#特征提取名都预训练网络名称的映射
name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_50_beta': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
    'xception_41': 'xception_41',
    'xception_65': 'xception_65',
    'xception_71': 'xception_71',
}

#像素平均值
_MEAN_RGB = [123.15, 115.90, 103.06]

def _preprocess_subtract_imagenet_mean(inputs):
    """减去 imagenet平均RGB值"""
    mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
    return inputs - mean_rgb

def _preprocess_zero_mean_unit_range(inputs):
    """将图像值从[0, 255]映射到[-1,1]"""
    return (2.0 /255.0) * tf.to_float(inputs) - 1.0

_PREPROCESS_FN = {
    'mobilenet_v2': _preprocess_zero_mean_unit_range,
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
    'xception_41': _preprocess_zero_mean_unit_range,
    'xception_65': _preprocess_zero_mean_unit_range,
    'xception_71': _preprocess_zero_mean_unit_range,
}

def mean_pixel(model_variant=None):
    """获得平均像素值
    此函数返回不同的平均像素值，具体取决于采用不同预处理函数的输入model_variant。 我们目前处理以下预处理功能：
   （1）_preprocess_subtract_imagenet_mean。 我们只返回平均像素值。
   （2）_preprocess_zero_mean_unit_range。 我们返回[127.5,127.5,127.5]。
   返回值的使用方式是预处理后的填充区域将包含值0。
    Args:
    :param model_variant:用于特征提取的模型变体。medel_variant=None时返回_MEAN_RGB.
    :return:
    平均像素值
    """
    if model_variant in ['resnet_v1_50',
                         'resnet_v1_101'] or model_variant is None:
        return _MEAN_RGB
    else:
        return  [127.5, 127.5, 127.5]


def get_network(network_name, preprocess_images, arg_scope=None):
    """获取网络

    :param network_name:网络名
    :param preprocess_images:是否图片预处理
    :param arg_scope: 可选，参数列表
    :return: 网络函数，用于提取特征
    :raises:
        ValueError 网络不支持
    """
    if network_name not in networks_map:
        raise ValueError('Unsupported network %s.'% network_name)
    arg_scope = arg_scope or arg_scopes_map[network_name]()
    def _identity_function(inputs):
        return inputs
    if preprocess_images:
        preprocess_function = _PREPROCESS_FN[network_name]
    else:
        preprocess_function = _identity_function
    func = networks_map[network_name]
    @functools.wraps(func)
    def network_fn(inputs, *args, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(preprocess_function(inputs), *args, **kwargs)
    return network_fn



def extract_features(images,
                     output_stride=8,
                     multi_grid=None,
                     depth_multiplier=1.0,
                     final_endpoint=None,
                     model_variant=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     regularize_depthwise=False,
                     preprocess_images=True,
                     num_classes=None,
                     global_pool=False):
    """通过特定的模型变体来提取特征

    Args:
    :param images:tensor,[batch, height, width, channels].
    :param output_stride: 输入输出空间分辨率的比值
    :param multi_grid: 在网络中使用不同的atrous率的层次结构。
    :param depth_multiplier:Float用于MobileNet中使用的所有卷积运算的深度（通道数）的浮点乘数。
    :param final_endpoint:用于构建网络的MobileNetz最后结束点。
    :param mode_variant:用于特征提取的模型变体
    :param weight_decay:用于模型变体的特征衰减
    :param reuse:是否复用模型变量
    :param is_training:是否训练
    :param fine_tune_batch_norm:是否微调ＢＮ参数
    :param regularize_depthwise:是否在depwise卷积权重上应用L2正则化
    :param preprocess_images:是否对图像执行预处理。 默认为True。 如果预处理将由其他函数完成，则设置为False。
    我们支持两种类型的预处理：（1）平均像素减法和（2）像素值归一化为[-1,1]。
    :param num_classes:图像分类任务的类数。 密集预测任务的默认值为None。
    :param global_pool:用于图像分类任务的全局池。 默认为False，因为密集预测任务不使用此功能。
    :return:
        features:tensor,[batch, feature height, feature width, feature_channels],
        feature的height和width由图片的height和width以及输出步长决定
        end_points:从网络组件到相应激活的字典。
    :raises:
    ValueError:无法识别的模型变体
    """
    if 'resnet' in model_variant:
        arg_scope = arg_scopes_map[model_variant](weight_decay=weight_decay,
                                                 batch_norm_decay=0.95,
                                                 batch_norm_epsilon=1e-5,
                                                 batch_norm_scale=True)
        features, end_points = get_network(
            model_variant, preprocess_images,arg_scope)(inputs=images,
                                                       num_classes=num_classes,
                                                       is_training=(is_training and fine_tune_batch_norm),
                                                       global_pool=global_pool,
                                                       output_stride=output_stride,
                                                       multi_grid=multi_grid,
                                                       reuse=reuse,
                                                       scope=name_scope[model_variant])
    elif 'xception' in model_variant:
        arg_scope = arg_scopes_map[model_variant](weight_decay=weight_decay,
                                                  batch_norm_decay=0.9997,
                                                  batch_norm_epsilon=1e-3,
                                                  batch_norm_scale=True,
                                                  regularize_depthwise=regularize_depthwise)
        features, end_points = get_network(
            model_variant, preprocess_images, arg_scope)(inputs=images,
                                                         num_classes=num_classes,
                                                         is_training=(is_training and fine_tune_batch_norm),
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         regularize_depthwise=regularize_depthwise,
                                                         multi_grid=multi_grid,
                                                         reuse=reuse,
                                                         scope=name_scope[model_variant])
    elif 'mobilenet' in model_variant:
        arg_scope = arg_scopes_map[model_variant](is_training=(is_training and fine_tune_batch_norm),
                                                  weight_decay=weight_decay)
        features, end_points = get_network(
            model_variant, preprocess_images, arg_scope)(inputs=images,
                                                         depth_multiplier=depth_multiplier,
                                                         output_stride=output_stride,
                                                         reuse=reuse,
                                                         scope=name_scope[model_variant],
                                                         final_endpoint=final_endpoint)
    else:
        raise ValueError('Unknown model variant %s.' % model_variant)

    return features, end_points

