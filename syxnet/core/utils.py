import  tensorflow as tf
import  tensorflow.contrib.slim as slim

def scale_dimension(dim, scale):
    """Scales the input dimension.

    Args:
    :param dim: 输入维度,Input dimension(标量或标量的 Tensor)
    :param scale: 应用在输入的标量的总量
    :return: 缩放维度
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0)* scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale +1.0)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """将一个可分离的二维卷积分成depthwise卷积和pointwise卷积

    Args:
    :param inputs:输入张量[batch, height, width, channels]
    :param filters: 1×1pointwise 卷积里的过滤器数量
    :param kernel_size: 长度为2的列表,过滤器的[kernel_height, kernel_width].如果两个值相同可以为整数
    :param rate:用于depthwise卷积的Atrous 卷积率
    :param weight_decay:用于调整模型的权重衰减
    :param depthwise_weights_initializer_stddev:用于depthwise卷积的截断的正常权重初始化器的标准偏差。
    :param pointwise_weights_initializer_stddev:用于pointwise卷积的截断的正常权重初始化器的标准偏差
    :param scope:操作的可选作用域

    :return:在分离可分离二维卷积后的计算的特征
    """
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size=kernel_size,
                                    depth_multiplier=1,
                                    rate=rate,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=depthwise_weights_initializer_stddev),
                                    weights_regularizer=None,
                                    scope=scope + '_depthwise')
    return  slim.conv2d(outputs,
                        filters,
                        1,
                        weights_initializer=tf.truncated_normal_initializer(stddev=pointwise_weights_initializer_stddev),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        scope=scope + '_pointwise')
