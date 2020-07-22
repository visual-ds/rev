import tensorflow as tf
import tf_slim as slim

def vgg_basenet(inputs, fatness = 64, dilation = True):
    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        # Block1
        net = slim.repeat(inputs, 2, slim.conv2d, fatness, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['pool1'] = net
        
        
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, fatness * 2, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        
        
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 4, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv4')
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        end_points['pool4'] = net
        
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        end_points['pool5'] = net

        # fc6 as conv, dilation is added
        if dilation:
            net = slim.conv2d(net, fatness * 16, [3, 3], rate=6, scope='fc6')
        else:
            net = slim.conv2d(net, fatness * 16, [3, 3], scope='fc6')
        end_points['fc6'] = net

        # fc7 as conv
        net = slim.conv2d(net, fatness * 16, [1, 1], scope='fc7')
        end_points['fc7'] = net

    return net, end_points;    



class PixelLinkNet(object):
    def __init__(self, inputs, is_training):


        self.feat_layers = ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
        self.weight_decay = 0.0005
        self.dropout_ratio = 0
        self.num_classes = 2
        self.num_neighbours = 8

        self.inputs = inputs
        self.is_training = is_training
        self.build_network()
        self.fuse_feat_layers()
        self.logits_to_scores()

    
    def logits_to_scores(self):
        self.pixel_cls_scores = tf.nn.softmax(self.pixel_cls_logits)
        self.pixel_cls_logits_flatten = self.flat_pixel_cls_values(self.pixel_cls_logits)
        self.pixel_cls_scores_flatten = self.flat_pixel_cls_values(self.pixel_cls_scores)

        shape = tf.shape(input=self.pixel_link_logits)

        self.pixel_link_logits = tf.reshape(self.pixel_link_logits, [shape[0], shape[1], shape[2], self.num_neighbours, 2])
        self.pixel_link_scores = tf.nn.softmax(self.pixel_link_logits)
        self.pixel_pos_scores = self.pixel_cls_scores[:, :, :, 1]
        self.link_pos_scores = self.pixel_link_scores[:, :, :, :, 1]

    def flat_pixel_cls_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape = [shape[0], -1, shape[-1]])
        return values

    def build_network(self):
        with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=tf.keras.regularizers.l2(0.5 * (self.weight_decay)),
                        weights_initializer= tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                        biases_initializer = tf.compat.v1.zeros_initializer()):
        
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as sc:
                self.arg_scope = sc
                self.net, self.end_points = vgg_basenet(inputs =  self.inputs)
    
    def fuse_feat_layers(self):
        self.pixel_cls_logits = self.fuse_by_cascade_conv1x1_upsample_sum(self.num_classes, scope = 'pixel_cls')
        self.pixel_link_logits = self.fuse_by_cascade_conv1x1_upsample_sum(self.num_neighbours * 2, scope = 'pixel_link')
    

    def upscore_layer(self, layer, target_layer):   
        target_shape = tf.shape(input=target_layer)[1:-1]
        upscored = tf.image.resize(layer, target_shape)
        return upscored 

    def score_layer(self, input_layer, num_classes, scope):
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1], 
                    stride=1,
                    activation_fn=None, 
                    scope='score_from_%s'%scope,
                    normalizer_fn=None)
            try:
                use_dropout = self.dropout_ratio > 0
            except:
                use_dropout = False
                
            if use_dropout:
                if not self.is_training:
                    self.dropout_ratio = 0
                keep_prob = 1.0 - self.dropout_ratio
                tf.compat.v1.logging.info('Using Dropout, with keep_prob = %f'%(keep_prob))
                logits = tf.nn.dropout(logits, 1 - (keep_prob))
        return logits

    def fuse_by_cascade_conv1x1_upsample_sum(self, num_classes, scope):
        num_layers = len(self.feat_layers)
        with tf.compat.v1.variable_scope(scope):
            smaller_score_map = None
            for idx in range(0, len(self.feat_layers))[::-1]:
                current_layer_name = self.feat_layers[idx]
                current_layer = self.end_points[current_layer_name]
                current_score_map = self.score_layer(current_layer, num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    upscore_map = self.upscore_layer(smaller_score_map, current_score_map)
                    smaller_score_map = current_score_map + upscore_map
        
        return smaller_score_map