"""与输入预处理相关的通用的函数"""
import  tensorflow as tf

def flip_dim(tensor_list, prob=0.5, dim=1):
    """随机从给定tensor里翻转一个维度

    随机翻转“Tensors”的决定是一起做的。 换句话说，传入的所有图像都没有被翻转。
    注：没有 tf.random_flip_left_right 和 tf.random_flip_up_down, 所以我们可以控制概率，并确保在图像上应用相同的决策。
    Args:
        tensor_list: 和维度有同样数目的‘Tensor’列表
        prob:left-right 翻转的概率
        dim:翻转的维度,0,1,...

    Returns:
        outputs:可能翻转的“Tensors”的列表以及末尾的指示符“Tensor”，如果输入被翻转则值为“True”，否则为“False”。

    Raises:
        ValueError:如果dim是负数或者比‘Tensor’的维度大
    """
    random_value = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension')
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """用给定的pad_value补给定的图片

    Args:
    :param image:3维tensor,形状为[height, width, channels]
    :param offset_height: 要添加到顶部的0的行数
    :param offset_width:要添加到左边的0的列数
    :param target_height:输出的高度
    :param target_width:输出的宽度
    :param pad_value:补图片张量的值

    Return:
    :return:3维张量,形状为[target_height, target_width, channels].

    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or target_ * arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                  ['Wrong image tensor rank [Expected][Actual]',
                                   3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
        image -= pad_value
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(tf.greater_equal(target_width, width),
                                    ['target_width must be >= width'])
    target_height_assert = tf.Assert(tf.greater_equal(target_height, height),
                                     ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
        after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
        after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(tf.logical_and(tf.greater_equal(after_padding_width,0),
                                             tf.greater_equal(after_padding_height,0)),
                              ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(image, paddings)
    return padded + pad_value

def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """使用提供的offsets 和尺寸来裁剪给定图片
    注：这一方法不能确保我们知道输入图像的尺寸,但它能确保我们知道输入图像范围

    Args:
    :param image:一张图片,形状为[height, width, channels]
    :param offset_height: 标量张量，表示高度偏移。
    :param offset_width:标量张量，表示宽度偏移。
    :param crop_height:裁剪图片的高度
    :param crop_width:裁剪图片的宽度

    Returns:
    :return:裁剪后的图片

    Raises:
    ValueError: 如果'image'的输入没有三个
    InvalidArgumentError:如果范围不是3或者图片的维度小于裁剪尺寸
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise  ValueError('input must have rank of 3')

    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3),
                               ['Rank of image must be equal to 3'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    #使用tf.slice 作为定义裁剪尺寸的接受tensors,而不是crop_to_bounding box
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image

def random_crop(image_list, crop_height, crop_width):
    """裁剪给定的图片列表

    Args
    :param image_list:有相同维度的图片tensors的列表但是通道可能不同
    :param crop_height: 新的高度
    :param crop_width: 新的宽度

    Ｒeturn:
    :return:裁剪后的图片列表

    Raise:
    ValueError: 如果有多张不一样大小的输入图片或者输入图片小于裁剪维度
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    #计算rank assertion
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                   ['Wrong rank for tensor %s [expected] [actual]',
                                    image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height),
                                                tf.greater_equal(image_width, crop_width)),
                                 ['Crop size greater than the image size.'])
    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(tf.equal(height, image_height),
                                  ['Wrong height for tensor %s [expected][actual]',
                                   image.name, height, image_width])
        width_assert = tf.Assert(tf.equal(width, image_width),
                                 ['Wrong width for tensor %s [expected][actual]'])
        asserts.extend([height_assert, width_assert])

    #创建一个随机边界框
    #使用tf.random_uniform 而不是numpy.random.rand 因为tf.random_uniform能在图片平均时间生成随机数字，
    #而numpy.random.rand在图像定义时产生随机数
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height +1, [])
        max_offset_width = tf.reshape(image_width - crop_width +1, [])
    offset_height = tf.random_uniform([],
                                      maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([],
                                     maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]

def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """获得一个随机尺度值

    Ａrgs:
    :param min_scale_factor:尺度最小值
    :param max_scale_factor: 尺度最大值
    :param step_size: 从最小到最大的步长尺寸

    :return: 在最大最小值之间选择一个可选的随机尺度

    Ｒaises:
    ValueError:min_scale_factor 有不期望的值
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)

    #如果step_size=0，均匀采样值从[min,max]
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    #当step_size !=0时,我们随机选取一个离散的值
    num_steps = int((max_scale_factor - min_scale_factor)/step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]

def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """随机扩展图片和标签

    Args:
    :param image:图片,尺寸为[height, width, 1]
    :param label: 标签,尺寸为[height,width, 1]
    :param scale:扩展图片和标签的值

    :return:扩展后的图片和标签
    """
    if scale == 1.0:
        return image,label
    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    #因为图片插值所以需要压缩和扩张维度, ４维tensor 作为输入
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0),
                                                new_dim,
                                                align_corners=True), [0])
    if label is not None:
        label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0),
                                                            new_dim,
                                                            align_corners=True), [0])
    return image, label

def resolve_shape(tensor, rank=None, scope=None):
    """完全地分解tensor的形状

    Ａrgs:
    :param tensor:输入张量
    :param rank:tensor的范围
    :param scope:可选命名范围
    :return:shape:张量的完整形状
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape

def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=tf.image.ResizeMethod.BILINEAR):
    """重新设置图片或标签的大小,使其边界在提供的范围内
    输出尺寸由两种情况：
     1.如果图像可以重新缩放以使其最小尺寸等于min_size而另一边不超过max_size，则执行此操作。
   　2.否则，调整大小以使最大边等于max_size。
    将`range（factor）`中的整数添加到计算边，使最终尺寸为'factor`加1的倍数。
    Args:
    :param image:3维tensor,[height, width, channels]
    :param label:(可选),一个三维tensor,[height, width, channels](default),或者[channels, height, width]当label_layout_is_chw =　True
    :param min_size:（标量）较小图像侧的所需大小。
    :param max_size:（标量）较大图像侧的最大允许大小。 请注意，输出维度不大于max_size，并且当factor不是None时，可能略小于min_size。
    :param factor:使输出大小倍数因子加1。
    :param align_corners:如果为True，则精确对齐输入和输出的所有4个角。
    :param label_layout_is_chw:如果为true，则标签的形状为[通道，高度，宽度]。我们支持这种情况，
    因为对于某些实例分割数据集，实例分割保存为[num_instances，height，width]。
    :param scope:可选名称范围。
    :param method:图像调整大小的方法。 默认为tf.image.ResizeMethod.BILINEAR

    :return:3Dtensor,[new_height, new_width, channels],
    min(new_height, new_width) == ceil(min_size)
    max(new_height, mew_width) == ceil(max_size)

    Raises:
    ValueError: 如果图片不是3D tensor
    """
    with tf.name_scope(scope, 'resize_to_range', [image]):
        new_tensor_list = []
        min_size = tf.to_float(min_size)
        if max_size is not None:
            max_size = tf.to_float(max_size)
            #将max_size修改为因子的倍数加1，并确保调整大小后的最大尺寸不大于max_size。
            if factor is not  None:
                max_size = (max_size + (factor - (max_size - 1) % factor) % factor - factor)
        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.to_float(orig_height)
        orig_width = tf.to_float(orig_width)
        orig_min_size = tf.minimum(orig_height, orig_width)

        #计算可能的更大尺寸
        large_scale_factor = min_size / orig_min_size
        large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
        large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
        large_size = tf.stack([large_height, large_width])

        new_size = large_size
        if max_size is not None:
            #计算可能的更小尺寸，当更大的过大时使用
            orig_max_size = tf.maximum(orig_height, orig_width)
            small_scale_factor = max_size / orig_max_size
            small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
            small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
            small_size = tf.stack([small_height, small_width])
            new_size = tf.cond(tf.to_float(tf.reduce_max(large_size)) > max_size,
                               lambda : small_size,
                               lambda : large_size)
            #确保两个输出边都是因子加1的倍数。
        if factor is not  None:
            new_size += (factor - (new_size - 1) % factor) % factor
        new_tensor_list.append(tf.image.resize_images(image,
                                                      new_size, method=method, align_corners=align_corners))
        if label is not None:
            if label_layout_is_chw:
                #输入标签有[channel, height, width]
                resized_label = tf.expand_dims(label, 3)
                resized_label = tf.image.resize_nearest_neighbor(resized_label, new_size, align_corners=align_corners)
                resized_label = tf.squeeze(resized_label, 3)
            else:
                #输入标签为[height, width, channels]
                resized_label = tf.image.resize_images(
                    label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    align_corners=align_corners)
            new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)

        return new_tensor_list



