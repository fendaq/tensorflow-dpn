#!/usr/bin/python
# encoding: utf-8

import tensorflow as tf
import numpy as np

import model_build
import model_utils

tf.app.flags.DEFINE_string("vgg_param_path", "../model/vgg16.npy", "file stores the paramters of vgg16 net")
tf.app.flags.DEFINE_integer("number_classes", 21, "total classes of the dataset + background")

flags = tf.app.flags.FLAGS

def train():
    dpn = model_build.DeepParseNet(flags.vgg_param_path)
    dpn.check()


class Trainer(object):
    """Trainer control the training process of this model."""
    def __init__(self, nc):
        self.net = model_build.DeepParseNet(flags.vgg_param_path)
        # TODO (meijieru) : add saver to the trainer
        #  self.saver =

    def b11(self, config):
        """Fine-tune b1 to b11 without the last four groups.

        Args:
            config: Control the epoch and the learning rate

        Returns:
            None

        """
        pass

    def b12(self, config):
        """Stack b12 on top of b11 and update its parameters.
        The weights of the preceding groups (i.e. b1~b11) are fixed.

        Args:
            config: Control the epoch and the learning rate

        Returns:
            None

        """
        pass

    def b13_b14(self, config):
        """Stack b13 and b14 on top of b12 and update its parameters.
        The weights of the preceding groups (i.e. b1~b12) are fixed.

        Args:
            config: Control the epoch and the learning rate

        Returns:
            None

        """
        pass

    def fine_tuned(self, config):
        """Fine-tuned all the paramters

        Args:
            config: Control the epoch and the learning rate

        Returns:
            None

        """
        pass

    def set_saver(saver):
        pass

    def save(self):
        pass



if __name__ == "__main__":
    trainer = Trainer(flags.number_classes)
    trainer.b11()
    trainer.b12()
    trainer.b13_b14()
    trainer.fine_tuned()
    trainer.save()
