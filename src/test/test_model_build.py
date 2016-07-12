#!/usr/bin/python
# encoding: utf-8

import unittest
import sys
import tensorflow as tf
import numpy as np

origin_path = sys.path
sys.path.append("..")
import model_build
import model_utils
sys.path = origin_path


def _suite():
    suite = unittest.TestSuite()
    suite.addTest(ModuleBuileTestCase("test_cross_min_pool_layer"))
    return suite


class ModuleBuileTestCase(unittest.TestCase):
    """ Test the ability of some layers. """
    def setUp(self):
        self.sess = tf.InteractiveSession()
        self.net = model_build.DeepParseNet()

    def tearDown(self):
        self.sess.close()
        del self.sess

    def test_cross_min_pool_layer(self):
        _bottom = np.asarray(range(0, 10))
        _bottom = np.reshape(_bottom, [1, 1, 1, 10])
        bottom = tf.Variable(_bottom)
        model_utils.reload_var(bottom, _bottom)
        layer = self.net.cross_min_pool_layer(bottom, 5, 2)
        output, = self.sess.run([layer])
        output = np.reshape(output, [5])
        self.assertEqual(output.tolist(), range(0, 10, 2))

if __name__ == "__main__":
    suite = _suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)
