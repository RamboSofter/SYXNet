"""提供模型定义和帮助函数
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from syxnet.core import dense_prediction_cell
from syxnet.core import feature_extractor
from syxnet.core import utils


LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

scale_dimension = utils.scale_dimension
split_separable_conv2d = utils.split_separable_conv2d

def get_extra_layer_scopes(last_layers_contain_logits_only=False):
  """从额外层获取参数

  :param last_layers_contain_logits_only:布尔值，为真则只包含分对数作为最后一层
  :return:一组额外层的列表
  """
  if last_layers_contain_logits_only:
      return [LOGITS_SCOPE_NAME]
  else:
      return [
          LOGITS_SCOPE_NAME,
          IMAGE_POOLING_SCOPE,
          ASPP_SCOPE,
          CONCAT_PROJECTION_SCOPE,
          DECODER_SCOPE,
          META_ARCHITECTURE_SCOPE,
      ]


def predict_labels_multi_scale(images,
                               model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
  """预测分割标签

  :param images: tensor,[batch, height, width, channels]
  :param model_options: 卷积网络模型选择来配置模型
  :param eval_scales: 用于调整图像大小以进行评估的比例
  :param add_flipped_images:是否添加翻转图像用于评估
  :return:具有指定output_type（例如，语义预测）的键的字典和存储表示预测的张量的值（通道上的argmax）.
  每个预测的大小[批次，高度，宽度]。
  """
  outputs_to_predictions = {
      output: []
      for output in model_options.outputs_to_num_classes
  }
  for i, image_scale in enumerate(eval_scales):
      with tf.variable_scope(tf.get_variable_scope(), reuse=True if i else None):
          outputs_to_scales_to_logits = multi_scale_logits(images,
                                                           model_options=model_options,
                                                           image_pyramid=[image_scale],
                                                           is_training=False,
                                                           fine_tune_batch_norm=False)
      if add_flipped_images:
          with tf.variable_scope(tf.get_variable_scope(), reuse=True):
              outputs_to_scales_to_logits_reversed = multi_scale_logits(
                  tf.reverse_v2(images, [2]),
                  model_options=model_options,
                  image_pyramid=[image_scale],
                  is_training=False,
                  fine_tune_batch_norm=False)
      for output in sorted(outputs_to_scales_to_logits):
          scales_to_logits = outputs_to_scales_to_logits[output]
          logits = tf.image.resize_bilinear(scales_to_logits[MERGED_LOGITS_SCOPE],
                                            tf.shape(images)[1:3],
                                            align_corners=True)
          outputs_to_predictions[output].append(
              tf.expand_dims(tf.nn.softmax(logits), 4))

      if add_flipped_images:
          scales_to_logits_reversed = (outputs_to_scales_to_logits_reversed[output])
          logits_reversed = tf.image.resize_bilinear(
              tf.reverse_v2(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
              tf.shape(images)[1:3],
              align_corners=True)
          outputs_to_predictions[output].append(
              tf.expand_dims(tf.nn.softmax(logits_reversed), 4))
  for output in  sorted(outputs_to_predictions):
      predictions = outputs_to_predictions[output]
      #计算不同尺度和翻转图像的平均预测。
      predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
      outputs_to_predictions[output] = tf.argmax(predictions, 3)
  return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
  """预测分割标签

  Args:
  :param images:tensor,[batch, height, width, channels].
  :param model_options: 卷积网络模型选择来配置模型
  :param image_pyramid: 输入图像尺度用于多尺度预测
  :return: 具有指定output_type（例如，语义预测）的键的字典和存储表示预测的张量的值（通道上的argmax）.
  每个预测的大小[批次，高度，宽度]。
  """
  outputs_to_scales_to_logits = multi_scale_logits(
      images,
      model_options=model_options,
      image_pyramid=image_pyramid,
      is_training=False,
      fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
      scales_to_logits = outputs_to_scales_to_logits[output]
      logits = tf.image.resize_bilinear(
          scales_to_logits[MERGED_LOGITS_SCOPE],
          tf.shape(images)[1:3],
          align_corners=True)
      predictions[output] = tf.argmax(logits, 3)

  return predictions


def _resize_bilinear(images, size, output_dtype=tf.float32):
  """返回调整大小后的图片作为输出类型
  Args:
  :param images:tensor, [batch, height_in, width_in, channels]
  :param size: 图像的新大小。1-D int32 Tensor of 2 elements：new_height,new_width。
  :param output_dtype:目标类型
  :return:更换为目标类型的tensor,[batch, height_out, width_out, channels]
  """
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
  """获得多尺度输入的logits
  对于训练和评估，返回的logits都被下采样（由于最大池化层）。

  :param images:tensor, [batch, height, width, channels]
  :param model_options:卷积网络模型选择来配置模型
  :param image_pyramid:用于多尺度特征提取的输入图像尺度
  :param weight_decay:模型变量的权重衰减
  :param is_training:是否训练
  :param fine_tune_batch_norm:是否微调BN的参数
  :return:
    outputs_to_scales_to_logits: 从output_type（例如，语义预测）到多尺度logits字典的映射到logits的映射。
    对于每个output_type，字典具有对应于与logits对应的比例和值的键。
    例如，如果`scales`等于[1.0,1.5]，那么键将包括'merged_logits'，'logits_1.00'和'logits_1.50'。
  :raises:
    ValueError:如果model_options未指定crop_size且其add_image_level_feature = True，
    则add_image_level_feature需要crop_size信息。
  """
  #设置默认值
  if not image_pyramid:
      image_pyramid = [1.0]
  crop_height = (
      model_options.crop_size[0]
      if model_options.crop_size else tf.shape(images)[1])
  crop_width = (
      model_options.crop_size[1]
      if model_options.crop_size else tf.shape(images)[2])
  #为输出分数计算height,width
  logits_output_stride = (
          model_options.decoder_output_stride or model_options.output_stride)
  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  #在图像金字塔中计算每个尺度的分数
  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }

  for image_scale in image_pyramid:
      if image_scale != 1.0:
          scaled_height = scale_dimension(crop_height, image_scale)
          scaled_width = scale_dimension(crop_width, image_scale)
          scaled_crop_size = [scaled_height, scaled_width]
          scaled_images = tf.image.resize_bilinear(images, scaled_crop_size, align_corners=True)
          if model_options.crop_size:
              scaled_images.set_shape([None, scaled_height, scaled_width, 3])
      else:
          scaled_crop_size = model_options.crop_size
          scaled_images = images

      updated_options = model_options._replace(crop_size=scaled_crop_size)
      outputs_to_logits = _get_logits(
          scaled_images,
          updated_options,
          weight_decay=weight_decay,
          reuse=tf.AUTO_REUSE,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm)

      #在融合前使得logits有相同的维度
      for output in sorted(outputs_to_logits):
          outputs_to_logits[output] = tf.image.resize_bilinear(
              outputs_to_logits[output], [logits_height, logits_width],
              align_corners=True)

      #当只有一个输入尺度时返回
      if len(image_pyramid) == 1:
          for output in sorted(model_options.outputs_to_num_classes):
              outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
          return outputs_to_scales_to_logits

      #给输出映射保存logits
      for output in sorted(model_options.outputs_to_num_classes):
          outputs_to_scales_to_logits[output]['logits_%.2f' % image_scale] = outputs_to_logits[output]

  # 从所有多尺度输入中融合logits
  for output in sorted(model_options.outputs_to_num_classes):
      #为每个输出类型级联多尺度ｌｏｇｉｔｓ
      all_logits = [tf.expand_dims(logits, axis=4)
                    for logits in outputs_to_scales_to_logits[output].values()]
      all_logits = tf.concat(all_logits, 4)
      merge_fn = (tf.reduce_max
                  if model_options.merge_method == 'max' else tf.reduce_mean)
      outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(all_logits, axis=4)

  return outputs_to_scales_to_logits


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
  """通过特定的模型变体提取特征
  Args:
  :param images:tensor, [batch, height, width, channels]
  :param model_options: 配置模型的模型选择
  :param weight_decay: 用于模型变量的权重衰减
  :param reuse: 是否重用模型变量
  :param is_training: 是否训练
  :param fine_tune_batch_norm: 是否微调BN

  :return:
    concat_logits: tensor,[batch, feature_height, feature_width,feature_channels],
    end_points: 从网络组件到相关激活的字典
  """
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)

  if not model_options.aspp_with_batch_norm:
      return features, end_points
  else:
      if model_options.dense_prediction_cell_config is not None:
          tf.logging.info('using dense prediction cell config.')
          dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
              config=model_options.dense_prediction_cell_config,
              hparams={
                  'conv_rate_multiplier': 16 // model_options.output_stride,
              })
          concat_logits = dense_prediction_layer.build_cell(
              features,
              output_stride=model_options.output_stride,
              crop_size=model_options.crop_size,
              image_pooling_crop_size=model_options.image_pooling_crop_size,
              weight_decay=weight_decay,
              reuse=reuse,
              is_training=is_training,
              fine_tune_batch_norm=fine_tune_batch_norm)
          return concat_logits, end_points
      else:
          #ASPP
          batch_norm_params = {
              'is_training': is_training and fine_tune_batch_norm,
              'decay': 0.9997,
              'epsilon': 1e-5,
              'scale': True,
          }
          with slim.arg_scope(
                  [slim.conv2d, slim.separable_conv2d],
                  weights_regularizer=slim.l2_regularizer(weight_decay),
                  activation_fn=tf.nn.relu,
                  normalizer_fn=slim.batch_norm,
                  padding='SAME',
                  stride=1,
                  reuse=reuse):
              with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                  depth = 256
                  branch_logits = []

                  if model_options.add_image_level_feature:
                      if model_options.crop_size is not None:
                          image_pooling_crop_size = model_options.image_pooling_crop_size
                          #如果image_pooling_crop_szie 没有特定,使用crop_size
                          if image_pooling_crop_size is None:
                              image_pooling_crop_size = model_options.crop_size
                          pool_height = scale_dimension(
                              image_pooling_crop_size[0],
                              1. / model_options.output_stride)
                          pool_width = scale_dimension(
                              image_pooling_crop_size[1],
                              1. / model_options.output_stride)
                          image_feature = slim.avg_pool2d(
                              features, [pool_height, pool_width], [1,1], padding='VALID')
                          resize_height = scale_dimension(
                              model_options.crop_size[0],
                              1. / model_options.output_stride)
                          resize_width = scale_dimension(
                              model_options.crop_size[1],
                              1. / model_options.output_stride)
                      else:
                          pool_height = tf.shape(features)[1]
                          pool_width = tf.shape(features)[2]
                          image_feature = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
                          resize_height = pool_height
                          resize_width = pool_width
                      image_feature = slim.conv2d(image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
                      image_feature = _resize_bilinear(image_feature,
                                                       [resize_height, resize_width],
                                                       image_feature.dtype)
                      #如果不是tensor　为resize_height/resize_width
                      if isinstance(resize_height, tf.Tensor):
                          resize_height = None
                      if isinstance(resize_width, tf.Tensor):
                          resize_width = None
                      image_feature.set_shape([None, resize_height, resize_width, depth])
                      branch_logits.append(image_feature)

                  branch_logits.append(slim.conv2d(features, depth, 1,
                                                   scope=ASPP_SCOPE + str(0)))
                  if model_options.atrous_rates:
                      #用有不同atrous率的3*3卷积
                      for i, rate in enumerate(model_options.atrous_rates, 1):
                          scope = ASPP_SCOPE + str(i)
                          if model_options.aspp_with_separable_conv:
                              aspp_features = split_separable_conv2d(features,
                                                                     filters=depth,
                                                                     rate=rate,
                                                                     weight_decay=weight_decay,
                                                                     scope=scope)
                          else:
                              aspp_features = slim.conv2d(features, depth, 3, rate=rate, scope=scope)
                          branch_logits.append(aspp_features)

                  #融合branch logits
                  concat_logits = tf.concat(branch_logits, 3)
                  concat_logits = slim.conv2d(
                      concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
                  concat_logits = slim.dropout(concat_logits,
                                               keep_prob=0.9,
                                               is_training=is_training,
                                               scope=CONCAT_PROJECTION_SCOPE + '_dropout')
                  return concat_logits, end_points


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False):
  """通过ASPP获得logits
  Args:
  :param images:tensor, [batch, height, width, channels]
  :param model_options: 配置模型
  :param weight_decay: 模型变量的权重衰减
  :param reuse: 是否重用模型变量
  :param is_training: 是否训练
  :param fine_tune_batch_norm:是否微调ＢＮ参数
  :return:
    outputs_to_logits:输出类型到logits的映射
  """
  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)
  if model_options.decoder_output_stride is not None:
      if model_options.crop_size is None:
          height = tf.shape(images)[1]
          width = tf.shape(images)[2]
      else:
          height, width = model_options.crop_size
      decoder_height = scale_dimension(height,
                                       1.0 / model_options.decoder_output_stride)
      decoder_width = scale_dimension(width,
                                      1.0 / model_options.decoder_output_stride)
      features = refine_by_decoder(features,
                                   end_points,
                                   decoder_height=decoder_height,
                                   decoder_width=decoder_width,
                                   decoder_use_separable_conv=model_options.decoder_use_separable_conv,
                                   model_variant=model_options.model_variant,
                                   weight_decay=weight_decay,
                                   reuse=reuse,
                                   is_training=is_training,
                                   fine_tune_batch_norm=fine_tune_batch_norm)
  outputs_to_logits = {}
  for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_logits[output] = get_branch_logits(features,
                                                    model_options.outputs_to_num_classes[output],
                                                    model_options.atrous_rates,
                                                    aspp_with_batch_norm=model_options.aspp_with_batch_norm,
                                                    kernel_size=model_options.logits_kernel_size,
                                                    weight_decay=weight_decay,
                                                    reuse=reuse,
                                                    scope_suffix=output)
  return outputs_to_logits


def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
  """添加解码器获得获得细化的分割结果

  :param features:tensor,[batch, features_height, features_width, features_channels]
  :param end_points:网络组件和相应激活的字典
  :param decoder_height:解码器特征图的高度
  :param decoder_width:解码器特征图的映射
  :param decoder_use_separable_conv:是否给解码器应用可分离卷积
  :param model_variant:用于特征提取的模型变体
  :param weight_decay:模型变量的权重衰减
  :param reuse:是否重用模型变量
  :param is_training:是否训练
  :param fine_tune_batch_norm:是否微调ＢＮ
  :return:
    解码器输出,[batch, decoder_height, decoder_width,decoder_width,decoder_channels]
  """
  batch_norm_params = {
      'is_training': is_training and fine_tune_batch_norm,
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
  }
  with slim.arg_scope(
          [slim.conv2d, slim.separable_conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          padding='SAME',
          stride=1,
          reuse=reuse):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
              feature_list = feature_extractor.networks_to_feature_maps[
                  model_variant][feature_extractor.DECODER_END_POINTS]
              if feature_list is None:
                  tf.logging.info('Not found any decoder end points.')
                  return features
              else:
                  decoder_features = features
                  for i, name in enumerate(feature_list):
                      decoder_features_list = [decoder_features]

                      #Mobilenet使用不同的命名规格
                      if 'mobilenet' in model_variant:
                          feature_name = name
                      else:
                          feature_name = '{}/{}'.format(feature_extractor.name_scope[model_variant], name)
                      decoder_features_list.append(slim.conv2d(end_points[feature_name],
                                                               48,
                                                               1,
                                                               scope='feature_projection'+str(i)))
                      #重新设置decoder_height/decoder_width大小
                      for j, feature in enumerate(decoder_features_list):
                          decoder_features_list[j] = tf.image.resize_bilinear(feature,
                                                                              [decoder_height, decoder_width],
                                                                              align_corners=True)
                          h = (None if isinstance(decoder_height, tf.Tensor)
                               else decoder_height)
                          w = (None if isinstance(decoder_width, tf.Tensor)
                               else decoder_width)
                          decoder_features_list[j].set_shape([None, h, w, None])
                          decoder_depth =256
                          if decoder_use_separable_conv:
                              decoder_features = split_separable_conv2d(
                                  tf.concat(decoder_features_list, 3),
                                  filters=decoder_depth,
                                  rate=1,
                                  weight_decay=weight_decay,
                                  scope='decoder_conv1')
                          else:
                              num_convs = 2
                              decoder_features = slim.repeat(
                                  tf.concat(decoder_features_list, 3),
                                  num_convs,
                                  slim.conv2d,
                                  decoder_depth,
                                  3,
                                  scope='decoder_conv' + str(i))
                  return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
  """获得每个模型分支的分数
  当采用atrous空间金字塔池时，底层模型在最后一层分支，并且所有分支被合并以形成最终logits。
  :param features:[batch, height, width, channels]
  :param num_classes:预测类别
  :param atrous_rates:最后一层的atrous卷积率列表
  :param aspp_with_batch_norm:对ASPP用BN
  :param kernel_size:卷积核大小
  :param weight_decay:用于模型变量的权重衰减
  :param reuse:是否重用模型变量
  :param scope_suffix:模型变量的视野下标
  :return:
    融合logits　[batch, height, width, num_classes]
  :raises:
    ValueError:无效的核大小值
  """
  # 当使用ASPP批量规范化时，之前在extract_features中应用了ASPP，因此我们只需在此处应用1x1卷积。
  if aspp_with_batch_norm or atrous_rates is None:
      if kernel_size != 1:
          raise  ValueError('Kernel sie must be 1 when atrous_rate is None or'
                            'using aspp_with_batch_norm. Gets %d.' % kernel_size)
      atrous_rates = [1]
  with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          reuse=reuse):
      with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
          branch_logits = []
          for i, rate in enumerate(atrous_rates):
              scope = scope_suffix
              if i:
                  scope += '_%d' % i
              branch_logits.append(slim.conv2d(features,
                                               num_classes,
                                               kernel_size=kernel_size,
                                               rate=rate,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               scope=scope))

          return tf.add_n(branch_logits)
