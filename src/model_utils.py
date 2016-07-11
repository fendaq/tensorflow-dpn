##
# @file model_utils.py
# @brief multiple useful utils
# @author meijieru, meijieru@gmail.com
# @version 0.1
# @date 2016-07-08


import tensorflow as tf
import numpy as np


def shape_probe(tensor):
    """ Get the shape of the tensor.

    Show the shape of the tensor when first evaluate it.

    Args:
        tensor: Target

    Returns:
        The same tensor as `tensor`.
    """
    return tf.Print(tensor, [tf.shape(tensor)],
                    message="shape of {} = ".format(tensor.op.name),
                    first_n=1)


def min_max_probe(tensor, first_n=1):
    """ Show the min & max value of the tensor.

    Useful for check the scale temporarily, for more information please use summaries.

    Args:
        tensor: Target
        first_n: Displaying for the first n times, `None` for every time

    Returns:
        The same tensor as `tensor`
    """
    return tf.Print(tensor, [tf.reduce_min(tensor), tf.reduce_max(tensor)],
                    message="Min Max = ", first_n=first_n)


def reload_var(var, array):
    """ Set the tensor to the value of array.

    Assuming that the default session have been lauched.

    Args:
        var: To be set
        array: Value for tensor, must has conincident size with the var

    Returns:
        None

    Raises:
        AssertionError: An error occured when the shapes are not compatible
    """
    assert (var.get_shape().as_list() == list(array.shape))
    assign_op = var.assign(array)
    assign_op.eval()


def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor

    Returns:
        None
    """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def reload_vgg(path):
    """Reload the paramters of the well trained vgg16 net

    Args:
        path: Path for the paramters' file

    Returns:
        None

    """

