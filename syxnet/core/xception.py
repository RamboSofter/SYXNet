"""Xception 模型
在MSRA上的改变：
1.全卷积:所有的最大池化层用步长为２的可分离卷积替换。这允许我们用atrous卷积在任意分辨率上提取特征图
2.受mobilenetV1的启发,在depthwise　卷积后添加Relu和BN
"""
import collections
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils
import tensorflow.contrib.slim as slim

_DEFAULT_MULTI_GRID = [1, 1, 1]

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """一个描述Xception块的命名元组。
    构成:
        scope:块名
        unit_fn:Xception单元函数，它将张量作为输入，并返回另一个张量与Xception单元的输出。
        args:长度等于块中单元数的列表。 该列表包含块中每个单元的一个字典，用作unit_fn的参数。

    """

def fixed_padding(inputs, kernel_size, rate=1):
    """沿着空间维度填充输入，与输入大小无关。

    Args
    :param inputs:tensor, [batch, height_in, width_in, channels]
    :param kernel_size:正整数,用于conv2d 和max_pool2d的核大小
    :param rate:atrous卷积率
    :return:
        output: 带输入的大小[batch，height_out，width_out，channels]的张量，无论完整（如果是kernel_size == 1）或填充（如果kernel_size> 1）。
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0,0]])
    return  padded_inputs

@slim.add_arg_scope
def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):
    """执行'SAME'填充的2-D可分离卷积
    如果stride >1 且use_explicit_padding 为真,则我们在带有'VALID'填充的2d卷积做明确的０填充
    注意：
    net = separable_conv2d_same(inputs, num_outputs, 3,
       depth_multiplier=1, stride=stride)

    相当于

     net = slim.separable_conv2d(inputs, num_outputs, 3,
       depth_multiplier=1, stride=1, padding='SAME')
     net = resnet_utils.subsample(net, factor=stride)

      但当输入的高度或宽度是偶数
    　net = slim.separable_conv2d(inputs, num_outputs, 3, stride=stride,
       depth_multiplier=1, padding='SAME')是不同的，这也是添加这个函数的原因
       因此，如果输入特征图具有偶数的高度或宽度，则设置“use_explicit_padding = False”会导致沿一个像素对应维度的特征错位。

    :param inputs:4-D张量,[batch, height_in, width_in, channels]
    :param num_outputs:Integer, 输出过滤器的数量
    :param kernel_size:Integer,过滤器的kernel_size
    :param depth_multiplier:每个输入通道的深度卷积输出通道的数量。 深度卷积输出通道的总数将等于“num_filters_in * depth_multiplier”。
    :param stride:Integer,输出步长
    :param rate:Integer,atrous卷积率
    :param use_explicit_padding:如果为真,使用特定的填充使得模型和开放资源版本完全吻合,否则使用Tensorflow的'SAME'填充
    :param regularize_depthwise:是否在depthwise 卷积权重上应用L2-norm正则化
    :param scope:范围
    :param kwargs:额外的关键参数运行slim.conv2d
    :return:
        output:卷积输出,4Dtensor,尺寸[batch, height_out, width_out, channels]
    """
    def _separable_conv2d(padding):
        """装饰separable conv2d"""
        return slim.separable_conv2d(inputs,
                                     num_outputs,
                                     kernel_size,
                                     depth_multiplier=depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     padding=padding,
                                     scope=scope,
                                     **kwargs)
    def _split_separable_conv2d(padding):
        """将separable conv2d 分成depthwise 和pointwise conv2d"""
        outputs = slim.separable_conv2d(inputs,
                                        None,
                                        kernel_size,
                                        depth_multiplier=depth_multiplier,
                                        stride=stride,
                                        rate=rate,
                                        padding=padding,
                                        scope=scope + '_depthwise',
                                        **kwargs)
        return slim.conv2d(outputs,
                           num_outputs,
                           1,
                           scope=scope + '_pointwise',
                           **kwargs)
    if stride == 1 or not use_explicit_padding:
        if regularize_depthwise:
            outputs = _separable_conv2d(padding='SAME')
        else:
            outputs = _split_separable_conv2d(padding='SAME')
    else:
        inputs = fixed_padding(inputs, kernel_size, rate)
        if regularize_depthwise:
            outputs = _separable_conv2d(padding='VALID')
        else:
            outputs = _split_separable_conv2d(padding='VALID')
    return outputs


@slim.add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None):
    """Xception 模型
    Xception 模型 = 'residual' + 'shortcut'.
    'residual'是三个可分离卷积得的特征。'shortcut'是由１*1卷积得到的特征。
    在某些情况下，“shortcut”路径可以是简单的标识函数，也可以不是（即没有shortcut）。
    在这里用具有步长的可分离卷积替代了原来的最大池化操作,因为在当前Tensorflow中,对atrous率没有很好地支持

    Args:
    :param inputs:tensor,[batch, height, width, channels].
    :param depth_list:三个整数的列表,指定一个Xception模块的深度值
    :param skip_connection_type:用于residual路径的跳跃连接的类型，只支持'conv','sum','none'
    :param stride:块单元的步长,决定下采样的效果,或单元输入与输出的比值
    :param unit_rate_list:三个整数的列表,决定Xception模块中每个可分离卷积的单元率
    :param rate:Integer,,atrous率
    :param activation_fn_in_separable_conv:在可分离卷积中是否包含激活函数
    :param regularize_depthwise:是否在depthwise卷积权重上应用Ｌ2正则化
    :param outputs_collections:Xception单元输出
    :param scope:可选
    :return:Xception模块的输出
    :raises
        ValueError:如果depth_list和unit_rate_list
    """
    if len(depth_list) != 3:
        raise ValueError('Except three elements in depth_list.')
    if unit_rate_list:
        if len(unit_rate_list) != 3:
            raise ValueError('Except three elements in unit_rate_list.')
    with tf.variable_scope(scope, 'xception_module', [inputs]) as sc:
        residual = inputs

        def _separable_conv(features, depth, kernel_size, depth_multiplier,
                            regularize_depthwise, rate, stride, scope):
            if activation_fn_in_separable_conv:
                activation_fn = tf.nn.relu
            else:
                activation_fn = None
                features = tf.nn.relu(features)
            return separable_conv2d_same(features,
                                         depth,
                                         kernel_size,
                                         depth_multiplier=depth_multiplier,
                                         stride=stride,
                                         rate=rate,
                                         activation_fn=activation_fn,
                                         regularize_depthwise=regularize_depthwise,
                                         scope=scope)

        for i in  range(3):
            residual = _separable_conv(residual,
                                       depth_list[i],
                                       kernel_size=3,
                                       depth_multiplier=1,
                                       regularize_depthwise=regularize_depthwise,
                                       rate=rate*unit_rate_list[i],
                                       stride=stride if i==2 else 1,
                                       scope='separable_conv' + str(i+1))
        if skip_connection_type == 'conv':
            shortcut = slim.conv2d(inputs,
                                   depth_list[-1],
                                   [1,1],
                                   stride=stride,
                                   activation_fn=None,
                                   scope='shortcut')
            outputs = residual + shortcut
        elif skip_connection_type == 'sum':
            outputs = residual + inputs
        elif skip_connection_type == 'none':
            outputs =residual
        else:
            raise ValueError('Unsupported skip connection type')
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                outputs)


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
    """堆叠Xception块并控制输出特征密度。
    1.这个函数形式上建立scopes用于Xception　或者'block_name/unit_1','block_name/unit_2'.etc
    2.这个函数允许用户明确地控制输出步长
    通过atrous卷积来控制输出特征密度
    Args:
    :param net:tensor,[batch, height, width, channels].
    :param blocks:和Xception块的数量相等的长度的列表
    :param output_stride:如果为空，，则输出将在虚设的网络步幅计算。
    如果output_stride不是空，则它指定所请求的输入与输出空间分辨率的比率，该比率需要等于从启动到某个Xception级别的单位步幅的乘积。
    例如，如果Xception使用步长为1,2,1,3,4,1的单位，则output_stride的有效值为1,2,6,24或None（相当于output_stride = 24）。
    :param outputs_collections:收集添加到Xception块的输出
    :return:等于特定输出步长的输出tensor

    :raises:
        ValueError:如果目标输出步长无效
    """
    #current_stride值保证函数的有效步长。这允许我们在应用下一个残余单元时调用atrous卷积将导致激活具有大于目标output_stride的步幅。
    current_stride = 1

    #atrous 卷积率参数
    rate = 1


    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not  None and current_stride > output_stride:
                    raise  ValueError('The target output_stride cannot be reached.')
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    #如果我们已达到目标output_stride，那么我们需要使用stride = 1的atrous卷积，
                    # 并将atrous rate乘以当前单位的步幅，以便在后续层中使用。
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached')

    return net

def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None):
    """Xception 模块的生成器
    此函数生成一系列Xception模型。 请参阅特定模型实例化的xception _ *（）方法，
    这些方法是通过选择生成不同深度Xception的不同块实例来获得的。
    Args:
    :param inputs:tensor,[batch, height_in, width_in, channels].必须是浮点数。 如果使用预训练检查点，则像素值应与训练期间相同.
    :param blocks:长度等于Xception块数的列表。 每个元素都是一个Xception Block对象，用于描述块中的单元。
    :param num_classes:分类任务的预测类数。如果为0或无，则返回logit层之前的特征。
    :param is_training:batch_norm层是否处于训练模式。
    :param global_pool:如果为True，我们在计算logits之前执行全局平均池。 设置为True表示图像分类，False表示密集预测。
    :param keep_prob:保留在log-logits dropout图层中使用的概率。
    :param output_stride:如果为空，，则输出将在虚设的网络步幅计算。
    如果output_stride不是空，则它指定所请求的输入与输出空间分辨率的比率，该比率需要等于从启动到某个Xception级别的单位步幅的乘积。
    :param reuse:是否应该重用网络及其变量。 必须给出能够重用“scope”。
    :param scope:可选'scope'
    :return:
        nets: 4级tensor[batch, heighy_out, width_out, channels_out].
        如果global_pool为False，则height_out和width_out与相应的height_in和width_in相比减少了output_stride因子，
        否则height_out和width_out都等于1。 如果num_classes为0或None，则net是最后一个Xception块的输出，
        可能在全局平均池之后。 如果num_classes是非零整数，则net包含pre-softmax激活。
        end_points: 网络相关函数的字典
    :raises:
        ValueError:如果目标output_stride 无效
    """
    with tf.variable_scope(
        scope,'xception', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + 'end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.separable_conv2d,
                             xception_module,
                             stack_blocks_dense,],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if output_stride is not None:
                    if output_stride % 2 != 0:
                        raise  ValueError('The output_stride needs to be a multiple of 2.')
                    output_stride /= 2
                #根块函数在输入上操作
                net = resnet_utils.conv2d_same(net, 32, 3, stride=2,
                                               scope='entry_flow/conv1_1')
                net = resnet_utils.conv2d_same(net, 64, 3, stride=1,
                                               scope='entry_flow/conv1_2')

                #提取特征用于entry_flow, middle_flow,exit_flow
                net = stack_blocks_dense(net, blocks, output_stride)

                #转换end_points_collection到end_points字典
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection, clear_collection=True)

                if global_pool:
                    #全局平均池化
                    net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                                       scope='prelogits_dropout')
                    net = slim.conv2d(net, num_classes, [1,1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
    """用于生成Xception块的帮助函数

    Args:
    :param scope:块名
    :param depth_list:瓶颈层用于各个单元的深度
    :param skip_connection_type: 跳跃连接的类型,只能是'conv', 'sum', 'none'
    :param activation_fn_in_separable_conv: 是否在可分离卷积中包含激活函数
    :param regularize_depthwise: 是否在depthwise卷积权重上应用L2-norm正则化
    :param num_units: 块中的单元数量
    :param stride: 块的步幅，作为最后一个单元的步幅实现。所有其他单位的步幅= 1。
    :param unit_rate_list:三个整数的列表，确定相应xception块中的单位速率。
    :return:一个Xception块
    """
    if unit_rate_list is None:
        unit_rate_list = _DEFAULT_MULTI_GRID
    return Block(scope, xception_module, [{'depth_list':  depth_list,
                                           'skip_connection_type': skip_connection_type,
                                           'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
                                           'regularize_depthwise': regularize_depthwise,
                                           'stride': stride,
                                           'unit_rate_list':unit_rate_list,}] * num_units)

def xception_41(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='Xception_41'):
    """Xception-41 模块"""
    blocks = [
        xception_block('entry_flow/block1',
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=8,
                       stride=1),
        xception_block('exit_flow/block1',
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return xception(inputs,
                    blocks=blocks,
                    num_classes=num_classes,
                    is_training=is_training,
                    global_pool=global_pool,
                    keep_prob=keep_prob,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope=scope)

def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_65'):
    """Xception-65"""
    blocks = [
        xception_block('entry_flow/block1',
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return xception(inputs,
                    blocks=blocks,
                    num_classes=num_classes,
                    is_training=is_training,
                    global_pool=global_pool,
                    keep_prob=keep_prob,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope=scope)

def xception_71(inputs,
                numclasses=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_71'):
    """Xception-71"""
    blocks = [
        xception_block('entry_flow/block1',
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block3',
                       depth_list=[256, 256 ,256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block4',
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=stack_blocks_dense,
                       num_units=1,
                       stride=1),
        xception_block('entry_flow/block5',
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('middle_flow/block1',
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),
        xception_block('exit_flow/block1',
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=multi_grid),
    ]
    return xception(inputs,
                    blocks=blocks,
                    num_classes=numclasses,
                    is_training=is_training,
                    global_pool=global_pool,
                    keep_prob=keep_prob,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope=scope)

def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):
    """定义xception 的默认参数
    Args
    :param weight_decay:用于正则化模型的权重衰减
    :param batch_norm_decay:在批量标准化中估计层激活统计时，移动平均衰减值
    :param batch_norm_epsilon:小批量常量，以防止在批量标准化中通过它们的方差归一化激活时除以零。
    :param batch_norm_scale:如果为True，则使用显式“gamma”乘数来缩放批量标准化层中的激活。
    :param weights_initializer_stddev:所述trunctated正常权重初始化的标准偏差
    :param activation_fn:Xception的激活函数
    :param regularize_depthwise:是否对深度卷积权重应用L2范数正则化。
    :param use_batch_norm:是否使用批量标准化。
    :return:用于Xception模型的`arg_scope`。
    """
    batch_norm_params = {
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale': batch_norm_scale,
    }
    if regularize_depthwise:
        depthwise_regularizer = slim.l2_regularizer(weight_decay)
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer = tf.truncated_normal_initializer(stddev=weights_initializer_stddev),
                        activation_fn = activation_fn,
                        normalizer_fn=slim.batch_norm if use_batch_norm else None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope(
                        [slim.separable_conv2d],
                        weights_regularizer=depthwise_regularizer) as arg_sc:
                    return arg_sc