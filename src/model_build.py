#!/usr/bin/python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import model_utils


class DeepParseNet(object):

    def __init__(self, path):
        if path:
            self.paramters = np.load(path).item()
            print("vgg params load complete")
        else:
            self.paramters = None

    def check_param(self):
        if self.paramters:
            for key, item in self.paramters.iteritems():
                print(key, item[0].shape, item[1].shape)

    def build(self, x, m, n, k, nc, scope="dpn", debug=False):
        with tf.variable_scope(scope + "_conv"):
            conv = self.conv_layer(x, 3, 64, 3, "SAME", True, scope="conv1_1")
            conv = self.conv_layer(
                conv, 64, 64, 3, "SAME", True, scope="conv1_2")
            conv = self.max_pooling_layer(conv, debug, name="pool1")
            conv = self.conv_layer(
                conv, 64, 128, 3, "SAME", True, scope="conv2_1")
            conv = self.conv_layer(
                conv, 128, 128, 3, "SAME", True, scope="conv2_2")
            conv = self.max_pooling_layer(conv, debug, name="pool2")
            conv = self.conv_layer(
                conv, 128, 256, 3, "SAME", True, scope="conv3_1")
            conv = self.conv_layer(
                conv, 256, 256, 3, "SAME", True, scope="conv3_2")
            conv = self.conv_layer(
                conv, 256, 256, 3, "SAME", True, scope="conv3_3")
            conv = self.max_pooling_layer(conv, debug, name="pool3")
            conv = self.conv_layer(
                conv, 256, 512, 3, "SAME", True, scope="conv4_1")
            conv = self.conv_layer(
                conv, 512, 512, 3, "SAME", True, scope="conv4_2")
            conv = self.conv_layer(
                conv, 512, 512, 3, "SAME", True, scope="conv4_3")
            conv = self.dilation_layer(conv, 512, 512, 3, 2,
                                       "SAME", True, scope="conv5_1")
            conv = self.dilation_layer(conv, 512, 512, 3, 2,
                                       "SAME", True, scope="conv5_2")
            conv = self.dilation_layer(conv, 512, 512, 3, 2,
                                       "SAME", True, scope="conv5_3")
            conv = self.dilation_layer(conv, 512, 4096, 7, 4,
                                       "SAME", True, scope="fc6")
            conv = self.conv_layer(conv, 4096, 4096, 1,
                                   "SAME", True, scope="fc7")
            conv = self.conv_layer(
                conv, 4096, nc, 1, "SAME", True, scope="classify")
        # default interpolation method is bilinear interpolation
        conv = tf.image.resize_images(conv, 512, 512)

        with tf.variable_scope(scope + "_crf"):
            crf = self.dist_layer(conv, m, scope="dist")
            crf = self.conv_layer(crf, nc, nc * k, n, "SAME", False, scope="mu")
            crf = self.cross_min_pool_layer(crf, k, scope="cross_pool")
            crf = self.combine_layer(conv, crf, scope="combine")

        return crf

    def conv_layer(self, bottom, n_in, n_out, k, p="SAME", bias=False, collection="conv", scope="conv"):
        with tf.variable_scope(scope) as scope:
            kernel = self.get_conv_filter(n_in, n_out, k, scope)
            tf.add_to_collection(collection, kernel)
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding=p)
            if bias:
                bias = self.get_bias(n_out, scope)
                tf.add_to_collection(collection, bias)
                conv = tf.nn.bias_add(conv, bias)
            conv = tf.nn.relu(conv)
        return conv

    def dilation_layer(self, bottom, n_in, n_out, k, rate, p="SAME", bias=False, collection="conv", scope="dilation"):
        with tf.variable_scope(scope) as scope:
            kernel = self.get_conv_filter(n_in, n_out, k, scope)
            tf.add_to_collection(collection, kernel)
            conv = tf.nn.atrous_conv2d(bottom, kernel, 2, padding=p)
            if bias:
                bias = self.get_bias(n_out, scope)
                tf.add_to_collection(collection, bias)
                conv = tf.nn.bias_add(conv, bias)
            conv = tf.nn.relu(conv)
        print(tf.shape(conv))
        return conv

    def dist_layer(self, bottom, m, scope="dist"):
        """Compute the b12 in dpn.

        This is the high order term of dpn which corresponds to a triple penalty,
        implying if (i, u) and (j, v) are compatible, then (i,u) should be also
        compatible with j’s nearby pixels (z, v), ∀z \in N_j
        $ \sum_{\forall z \in N_j} d(j,z) q_j^v q_z^v $

        Args:
            TODO (meijieru) : check the size of the bottom
            bottom: Input tensor with size [batch_size, weight, height, channel]
            m: The receptive field of the convolution

        Kwargs:
            scope: variable_scope for this layer's weights

        Returns:
            conv: Tensor implies the j and its nerghbor"s information

        """
        pass

    def cross_min_pool_layer(self, bottom, k, scope="cross_pool"):
        """Min pooling within k channel.

        Calculating the min value cross the k channel, corresponding to choose
        the minimum \mu_k(i, u, j, v).

        Args:
            bottom: Input tensor with size [batch_size, weight, height, channel]
            k: Number of components in mixture.

        Kwargs:
            scope: variable_scope for this layer's weights

        Returns:
            pool: Tensor which is the minimum cross k channel.

        """
        pass

    def max_pooling_layer(self, bottom, debug=False, name="max_pooling"):
        pool = tf.nn.max_pool(bottom, [1, 2, 2, 1], [
                              1, 2, 2, 1], padding="SAME", name=name)
        if debug:
            pool = model_utils.shape_probe(pool)
        return pool

    def combine_layer(self, unary, high_order, scope="combine"):
        """Combine the unary term and the high order term in CRF.

        $ w_{(i, u)} = exp[\ln(unary_{(i, u)}) - high\_order_{(i, u)}] $
        $ o_{(i, u)} = \frac{w_{(i, u)}} {\sum_{u=1}^{nc} w_{(i, u)}} $

        Args:
            unary: Unary term of CRF
            high_order: Smooth term of CRF

        Kwargs:
            scope: variable_scope for this layer's weights

        Returns:
            crf: Finally probability of the image

        """
        w = tf.exp(tf.log(unary) - high_order)
        crf = w / tf.reduce_sum(w, 3, True)
        return crf

    def get_bias(self, n_out, name):
        if self.paramters:
            bias_init = self.paramters[name][1]
            #  assert n_out == bias_init.shape
        else:
            bias_init = tf.zeros(n_out)
        return tf.Variable(bias_init, name="bias")

    def get_conv_filter(self, n_in, n_out, k, name):
        if self.paramters:
            kernel_init = self.paramters[name][0]
            if name is "fc6":
                kernel_init = np.reshape(kernel_init, [7, 7, 512, 4096])
            elif name is "fc7":
                kernel_init = np.reshape(kernel_init, [1, 1, 4096, 4096])
        else:
            kernel_init = tf.truncated_normal([k, k, n_in, n_out],
                                              stddev=math.sqrt(2 / (k * k * n_in))),
        return tf.Variable(kernel_init, name="weight")
