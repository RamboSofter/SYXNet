import tensorflow as tf
from collections import  OrderedDict
from keras.models import Sequential

class DenseASPP(tf.nn):
    """
    *Output_scales can only set as 8 or 16
    """
    def __init__(self, model_cfg, n_class=19, output_stride=8):
        super(DenseASPP, self).__init__()
        bn_size = model_cfg['bn_size']
        drop_rate = model_cfg['drop_rate']
        growth_rate = model_cfg['growth_rate']
        num_init_features = model_cfg['num_init_features']
        block_config = model_cfg['block_config']

        dropout0 = model_cfg['dropout0']
        dropout1 = model_cfg['dropout1']
        d_feature0 = model_cfg['d_feature0']
        d_feature1 = model_cfg['d_feature1']

        feature_size = int(output_stride / 8)

        #First convolution
        self.features = Sequential(OrderedDict([
            ('conv0', tf.nn.conv2d(3, num_init_features, kernel_size =7, strides=2, padding=3, bias=False))
            ('norm0', tf.nn.batch_normalization(num_init_features))
            ('relu0', tf.nn.relu(inplace= True))
            ('pool0', tf.nn.max_pool_grad_v2(ksize=3, strides=2, padding=1))
        ]))

        # Each denseblock
        num_features = num_init_features

        #block1
        block = _DenseBlock(num_layers = block_config[0],
                            num_input_features = num_features,
                            bn_size = bn_size,
                            growth_rate =growth_rate,
                            drop_rate = drop_rate)
        self.features.add('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features = num_features, num_output_features =num_features // 2)
        self.features.add('transition%d' % 1,trans)
        num_features = num_features // 2

        #block2
        block = _DenseBlock(num_layers =block_config[1],
                            num_input_features = num_features,
                            bn_size =bn_size,
                            growth_rate = growth_rate,
                            drop_rate =drop_rate)
        self.features.add('denseblock %d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features =num_features, num_output_features =num_features // 2, stride = feature_size)
        self.features.add('transition %d' % 2,trans)
        num_features = num_features // 2

        #block3
        block = _DenseBlock(num_layers=block_config[2],
                            num_input_features= num_features,
                            bn_size= bn_size,
                            growth_rate= growth_rate,
                            drop_rate= drop_rate,
                            dilation_rate= int(2 / feature_size))
        self.features.add('denseblock %d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features= num_features,
                            num_output_features= num_features // 2, stride= 1)
        self.features.add('transition%d' % 3, trans)
        num_features = num_features // 2


        #block4
        block = _DenseBlock(num_layers= block_config[3],
                            num_input_features= num_features,
                            bn_size= bn_size,
                            growth_rate= growth_rate,
                            drop_rate= drop_rate,
                            dilation_rate= int(4 / feature_size))
        self.features.add('denseblock%d' % 4, trans)
        num_features =num_features // 2

        #Final batch norm
        self.features.add('norm5', tf.nn.batch_normalization(num_features))
        if feature_size > 1:
            self.features.add('upsample', tf.Upsample(scale_factor = 2, mode ='bilinear'))

        self.ASPP_3 = _DenseAsppBlock(input_num= num_features,
                                      num1= d_feature0,
                                      num2= d_feature1,
                                      dilation_rate= 3,
                                      drop_out= dropout0,
                                      bn_start= True)
        self.ASPP_6 = _DenseAsppBlock(input_num= num_features + d_feature1 * 1,
                                      num1=d_feature0,
                                      num2= d_feature1,
                                      dilation_rate= 6,
                                      drop_out=dropout0,
                                      bn_start=True)
        self.ASPP_12 = _DenseAsppBlock(input_num= num_features + d_feature1 * 2,
                                       num1= d_feature0,
                                       num2= d_feature1,
                                       dilation_rate=12,
                                       drop_out= dropout0,
                                       bn_start=True)
        self.ASPP_18 = _DenseAsppBlock(input_num=  num_features +d_feature1 * 3,
                                       num1= d_feature0,
                                       num2= d_feature1,
                                       dilation_rate= 18,
                                       drop_out= dropout0,
                                       bn_start=True)
        self.ASPP_24 = _DenseAsppBlock(input_num= num_features + d_feature1 * 4,
                                       num1 = d_feature0,
                                       num2 = d_feature1,
                                       dilation_rate = 24,
                                       drop_out= dropout0,
                                       bn_start= True)
        num_features = num_features + 5 *d_feature1

        self.classification = Sequential(
            tf.nn.dropout(noise_shape= dropout1),
            tf.nn.conv2d(in_channels =num_features, out_channels = n_class, kernel_size = 1,
                         padding= 0),
            tf.nn.Upsample(scale_factor = 8, mode ='bilinear')
        )

        for m in self.modules():
            if isinstance(m, tf.nn.conv2d):
                tf.nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, tf.nn.batch_normalization):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        def forward(self, _input):
            feature = self.features(_input)

            aspp3 = self.ASPP_3(feature)
            feature = tf.concat((aspp3,feature), dim = 1)

            aspp6 = self.ASPP_6(feature)
            feature = tf.concat((aspp6, feature), dim = 1)

            aspp12 = self.ASPP_12(feature)
            feature = tf.concat((aspp12, feature), dim = 1)

            aspp18 = self.ASPP_18(feature)
            feature = tf.concat((aspp18, feature), dim = 1)

            aspp24 = self.ASPP_24(feature)
            feature = tf.concat((aspp24, feature), dim = 1)

            cls = self.classification(feature)

            return  cls










class _DenseAsppBlock(Sequential):
    """ConvNet block for building DenseASPP."""

    def __init__(self, input_num, num1, num2 , dilation_rate, drop_out, bn_start =True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add('norm.1', tf.nn.batch_normalization(input_num, momentum = 0.0003)),
            self.add('relu.1', tf.nn.relu(inplace = True)),
            self.add('conv.1', tf.nn.conv2d(in_channels = input_num, out_channels = num1, kernel_size = 1)),

            self.add('norm.2', tf.nn.batch_normalization(num1, momentum = 0.0003)),
            self.add('relu.2', tf.nn.relu(inplace =True)),
            self.add('conv.2', tf.nn.conv2d(in_channels = num1, out_channels = num2,
                                            dilations = dilation_rate,
                                            padding= dilation_rate))
            self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature =tf.nn.function.dropout2d(feature, p = self.drop_rate, training =self.training)

        return feature

class _DenseLayer(Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate = 1):
        super(_DenseLayer, self).__init__()
        self.add('norm.1', tf.nn.batch_normalization(num_input_features)),
        self.add('relu.1', tf.nn.relu(inplace =True)),
        self.add('conv.1', tf.nn.conv2d(num_input_features, bn_size * growth_rate, kernel_size = 1, strides=1, bias =False ))

        self.add('norm.2', tf.nn.batch_normalization(bn_size * growth_rate))
        self.add('relu.2', tf.nn.relu(inplace = True))
        self.add('conv.2', tf.nn.conv2d(bn_size * growth_rate, growth_rate,
                                        kernel_size =3, strides=1, dilations= dilation_rate, padding=dilation_rate,
                                        bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate >0 :
            new_features = tf.nn.dropout(new_features, p = self.drop_rate,
                                         training =self.taining)
        return  tf.concat([x, new_features], 1)


class _DenseBlock(Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate = dilation_rate)

class _Transition(Sequential):
    def __init__(self, num_input_features, num_output_features, stride = 2):
        super(_Transition, self).__init__()
        self.add('norm', tf.nn.batch_normalization(num_input_features))
        self.add('relu', tf.nn.relu(inplace=True))
        self.add('conv', tf.nn.conv2d(num_input_features, num_output_features, kernel_size=1, strides=1, bias=False))
        if stride == 2:
            self.add('pool', tf.nn.avg_pool(ksize=2, strides=stride))


if __name__ == "__main__" :
    model = DenseASPP(2)
    print(model)
