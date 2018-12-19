"""ResNet V1 模型变体
"""
from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function
import tensorflow as tf
import functools
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim.nets import  resnet_utils


_DEFAULT_MULTI_GRID = [1, 1, 1]

@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               unit_rate=1,
               rate=1,
               outputs_collections=None,
               scope=None):
    """卷积后BN的瓶颈残余单元变体。
    请注意，我们在这里使用瓶颈变体，它具有额外的瓶颈层。
    将两个连续的ResNet块放在一起使用时，应该在第一个块的最后一个单元中使用stride = 2。
    Args:
    :param inputs:tensor,[batch, height, width, channels]
    :param depth:ResNet单元的输出深度
    :param depth_bottleneck:瓶颈层的深度
    :param stride:ResNet单元的步长.确定与输入相比的单位输出的下采样量。
    :param unit_rate:Integer,用于atrous 卷积的单元率
    :param rate:Integer.atrous卷积率
    :param outputs_collections:用于添加ResNet单元输出的集合。
    :param scope:可选变量范围

    :return:ResNet单元的输出
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in =slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut =slim.conv2d(inputs,
                                  depth,
                                  [1,1],
                                  stride=stride,
                                  activation_fn=None,
                                  scope='shortcut')
        residual = slim.conv2d(inputs, depth_bottleneck, [1,1], stride=1, scope='conv1')

        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate*unit_rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1,1], stride=1,
                               activation_fn=None, scope='conv3')

        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)

def root_block_fn_for_beta_variant(net):
    """获取beta变体的root_block_fn。
    Args:
    :param net:tensor,模型的输入[batch, height, width, channels]
    :return:在３个3*3卷积后的tensor
    """
    net = resnet_utils.conv2d_same(net, 64, 3, stride=2, scope='conv1_1')
    net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
    net = resnet_utils.conv2d_same(net, 128, 3, stride=1, scope='conv1_3')

    return net

def resnet_v1_beta(inputs,
                   blocks,
                   num_classes=None,
                   is_training=None,
                   global_pool=True,
                   output_stride=None,
                   root_block_fn=None,
                   reuse=None,
                   scope=None):
    """V1 ResNet模型的生成器
    这个函数生成一系列经过修改的ResNet v1模型。 特别是，第一个原始7x7卷积被三个3x3卷积替换。
     有关特定模型实例的信息，请参阅resnet_v1 _ *（）方法，这些方法是通过选择生成不同深度ResNets的不同块实例来获得的。
    :param inputs:tensor,[batch, height_in, width_in, channels]
    :param blocks:长度等于ResNet块数的列表。 每个元素都是一个resnet_utils.Block对象，用于描述块中的单位。
    :param num_classes:用于分类任务的预测类数。如果为None在最后的回归层返回特征
    :param is_training:是否用批正则化
    :param global_pool:如果为True,在计算回归之前使用全局平均池化。设置为真用于图像分类,假则用于密集预测。
    :param output_stride:如果为None，则输出将在标称网络步长处计算。 如果output_stride不是None，则它指定所请求的输入与输出空间分辨率的比率。
    :param root_block_fn:该函数由应用于根输入的卷积运算组成。 如果root_block_fn为None，
    则使用RseNet-v1的原始设置，该设置只是一个带有7x7内核和stride = 2的卷积。
    :param reuse:是否应该重用网络及其变量。 重用的话，必须给定重用的'scope'
    :param scope:可选variable_scope

    :return:nets.等级-4张量的大小[batch，height_out，width_out，channels_out]。 如果global_pool为False，
    则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
    否则height_out和width_out都等于1。 如果num_classes为None，则net是最后一个ResNet块的输出，可能是在全局平均池之后。
    如果num_classes不是None，则net包含pre-softmax激活。
    end_points: 从网络组件到相应激活的字典。

    :raise:如果目标输出步长无效则
    """
    if root_block_fn is None:
        root_block_fn = functools.partial(resnet_utils.conv2d_same,
                                          num_outputs=64,
                                          kernel_size=7,
                                          stride=2,
                                          scope='conv1')
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc :
        end_point_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            output_collections=end_point_collection):
            if is_training is not None:
                arg_scope = slim.arg_scope([slim.batch_norm], is_training=is_training)
            else:
                arg_scope = slim.arg_scope([])
            with arg_scope:
                net =inputs
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride /= 4
                net = root_block_fn(net)
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)

                if global_pool:
                    #全局平均池化
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net =slim.conv2d(net, num_classes, [1,1], activation_fn=None,
                                     normalizer_fn=None, scope='logits')
                #将end_points_collection 转变成end_points 的字典

                end_points = slim.utils.convert_collection_to_dict(end_point_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v1_beta_block(scope, base_depth, num_units, stride):
    """辅助函数，用于创建resnet_v1 beta变量的瓶颈块

    Args:
    :param scope:块的范围
    :param base_depth:对于每个单元的瓶颈层的深度
    :param num_units:块中的单元数量
    :param stride:该块的步幅，作为最后一个单元的步幅实现。 所有其他单位的步幅= 1。
    :return:一个resnet_v1 瓶颈块
    """
    return resnet_utils.Block(scope, bottleneck, [{'depth': base_depth * 4,

                                                   'depth_bottleneck': base_depth,
                                                   'stride': 1,
                                                   }] * (num_units -1) +[{'depth': base_depth * 4,
                                                                          'depth_bottleneck': base_depth,
                                                                          'stride':stride,
                                                                          'unit_rate': 1}])

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=None,
                 global_pool=False,
                 output_stride=None,
                 multi_grid=None,
                 reuse=None,
                 scope='resnet_v1_50'):
    """Resnet v1 50.

    Args:
    :param inputs:tensor,[batch, height_in, width_in, channels]
    :param num_classes: 用于分类任务的预测类别数
    :param is_training: 是否用ＢＮ进行训练
    :param global_pool: 如果为真，则在计算logits前执行全局平均池化。真用于图像分类，假用于预测
    :param output_stride: 如果为None，则输出将在标称网络步长处计算。 如果output_stride不是None，则它指定所请求的输入与输出空间分辨率的比率。
    :param multi_grid:在网络中使用不同的atrous卷积率的层次结构
    :param reuse:是否应该重用网络及其变量。 重用的话，必须给定重用的'scope'
    :param scope:可选variable_scope

    :return:net:4级张量的大小[batch，height_out，width_out，channels_out]。 如果global_pool为False，
    则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
    否则height_out和width_out都等于1。 如果num_classes为None，则net是最后一个ResNet块的输出，可能是在全局平均池之后。
    如果num_classes不是None，则net包含pre-softmax激活。
    end_points: 从网络组件到相应激活的字典。
    :raise: ValueError: 如果multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid =_DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units = 3, stride=2),
              resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_beta_block('block3', base_depth=256, num_units=6, stride=2),
              resnet_utils.Block('block4', bottleneck, [{'depth': 2048,
                                                         'depth_bottleneck':512,
                                                         'stride': 1,
                                                         'unit_rate': rate}for rate in multi_grid]),]
    return resnet_v1_beta(inputs,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          reuse=reuse,
                          scope=scope)

def resnet_v1_50_beta(inputs,
                      num_classes=None,
                      is_training=None,
                      global_pool=False,
                      output_stride=None,
                      multi_grid=None,
                      reuse=None,
                      scope='resnet_v1_50'):
    """Resnet v1 50 beta 变体

    Args:
    :param inputs:tensor 尺寸,[batch, height_in, width, channels]
    :param num_classes: 用于分类任务的预测类别数
    :param is_training:是否用ＢＮ进行训练
    :param global_pool: 如果为真，则在计算logits前执行全局平均池化。真用于图像分类，假用于预测
    :param output_stride:如果为None，则输出将在标称网络步长处计算。 如果output_stride不是None，则它指定所请求的输入与输出空间分辨率的比率。
    :param multi_grid:在网络中使用不同的atrous卷积率的层次结构
    :param reuse:是否应该重用网络及其变量。 重用的话，必须给定重用的'scope'
    :param scope:可选variable_scope
    :return:net:4级张量的大小[batch，height_out，width_out，channels_out]。 如果global_pool为False，
    则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
    否则height_out和width_out都等于1。 如果num_classes为None，则net是最后一个ResNet块的输出，可能是在全局平均池之后。
    如果num_classes不是None，则net包含pre-softmax激活。
    end_points: 从网络组件到相应激活的字典。
    :raise ValueError:如果multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) !=3:
            raise  ValueError('Expect multi_grid to have length 3.')

    blocks =[resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2),
             resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2),
             resnet_v1_beta_block('block3', base_depth=256, num_units=6, stride=2),
             resnet_utils.Block('block4', bottleneck, [{'depth': 2048,
                                                        'depth_bottleneck':512,
                                                        'stride':1,
                                                        'unit_rate':rate} for rate in multi_grid]),]
    return resnet_v1_beta(inputs,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          root_block_fn=functools.partial(root_block_fn_for_beta_variant),
                          reuse=reuse,
                          scope=scope)

def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=False,
                  output_stride=None,
                  multi_grid=None,
                  reuse=None,
                  scope='resnet_v1_101'):
    """Resnet v1 101

    Args:
    :param inputs: tensor,[batch, height_in, width_in, channels]
    :param num_classes: 用于分类任务的预测类别数
    :param is_training: 是否用ＢＮ进行训练
    :param global_pool: 如果为真，则在计算logits前执行全局平均池化。真用于图像分类，假用于预测
    :param output_stride: 如果为None，则输出将在标称网络步长处计算。 如果output_stride不是None，则它指定所请求的输入与输出空间分辨率的比率。
    :param multi_grid: 在网络中使用不同的atrous卷积率的层次结构
    :param reuse: 是否应该重用网络及其变量。 重用的话，必须给定重用的'scope'
    :param scope: 可选variable_scope
    :return: net:4级张量的大小[batch，height_out，width_out，channels_out]。 如果global_pool为False，
    则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
    否则height_out和width_out都等于1。 如果num_classes为None，则net是最后一个ResNet块的输出，可能是在全局平均池之后。
    如果num_classes不是None，则net包含pre-softmax激活。
    end_points: 从网络组件到相应激活的字典。
    :raise ValueError:如果multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_beta_block('block2', base_depth=128, num_units=3, stride=2),
              resnet_v1_beta_block('block3', base_depth=256, num_units=23, stride=2),
              resnet_utils.Block('block4', bottleneck, [{'depth': 2048,
                                                         'depth_bottleneck':512,
                                                         'stride':1,
                                                         'unit_rate': rate} for rate in multi_grid]),]
    return resnet_v1_beta(inputs,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          reuse=reuse,
                          scope=scope)

def resnet_v1_101_beta(inputs,
                       num_classes=None,
                       is_training=None,
                       global_pool=False,
                       output_stride=None,
                       multi_grid=None,
                       reuse=None,
                       scope='resnet_v1_101'):
    """Resnet v1 101 beta变体

    :param inputs:tensor 尺寸,[batch, height_in, width, channels]
    :param num_classes: 用于分类任务的预测类别数
    :param is_training:是否用ＢＮ进行训练
    :param global_pool: 如果为真，则在计算logits前执行全局平均池化。真用于图像分类，假用于预测
    :param output_stride:如果为None，则输出将在标称网络步长处计算。 如果output_stride不是None，则它指定所请求的输入与输出空间分辨率的比率。
    :param multi_grid:在网络中使用不同的atrous卷积率的层次结构
    :param reuse:是否应该重用网络及其变量。 重用的话，必须给定重用的'scope'
    :param scope:可选variable_scope
    :return:net:4级张量的大小[batch，height_out，width_out，channels_out]。 如果global_pool为False，
    则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
    否则height_out和width_out都等于1。 如果num_classes为None，则net是最后一个ResNet块的输出，可能是在全局平均池之后。
    如果num_classes不是None，则net包含pre-softmax激活。
    end_points: 从网络组件到相应激活的字典。
    :raise ValueError:如果multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3 :
            raise ValueError('Expect multi_grid to have length 3.')
    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_beta_block('block3', base_depth=256, num_units=23, stride=2),
              resnet_utils.Block('block4', bottleneck, [{'depth' :2048,
                                                         'depth_bottleneck': 512,
                                                         'stride': 1,
                                                         'unit_rate': rate} for rate in multi_grid]),]
    return resnet_v1_beta(inputs,
                          blocks=blocks,
                          num_classes=num_classes,
                          is_training=is_training,
                          global_pool=global_pool,
                          output_stride=output_stride,
                          root_block_fn=functools.partial(root_block_fn_for_beta_variant),
                          reuse=reuse,
                          scope=scope)

