#!/usr/bin/python
# encoding: utf-8

import unittest
import sys
import tensorflow as tf
import numpy as np

origin_path = sys.path
sys.path.append("..")
import model_utils
sys.path = origin_path


class ModelUtilsTestCase(unittest.TestCase):
    """ Test the ability of utils """

    def setUp(self):
        self.sess = tf.InteractiveSession()

    def runTest(self):
        def test_reload_var():
            a = tf.Variable(tf.constant(0, shape=[10]))

            init_op = tf.initialize_all_variables()
            self.sess.run(init_op)

            b = np.asarray([i for i in range(0, 10)])
            model_utils.reload_var(a, b)
            self.assertEqual(a.eval().tolist(), b.tolist())

        test_reload_var()

    def tearDown(self):
        self.sess.close()
        del self.sess


if __name__ == "__main__":
    unittest.main()
