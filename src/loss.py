#!/usr/bin/python
# encoding: utf-8

import tensorflow as tf


def loss(logits, labels, nc):
    """Calculate loss between prediction and labels

    Args:
        logits: Tensor with shape [batch, width, height, nc]
        labels: Groundtruth with the same size of logits
        nc: Number of classes

    Returns:
        loss: Loss tensor

    """
    with tf.name_scope("loss"):
        logits = tf.reshape(logits, [-1, nc])
        #  epsilon = tf.constant(1e-5)
        #  logits += epsilon
        labels = tf.to_float(tf.reshape(labels, [-1, nc]))

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name="loss")
        tf.add_to_collection("losses", cross_entropy_loss)
        loss = tf.add_n(tf.get_collection("losses", name="total loss"))

    return loss
