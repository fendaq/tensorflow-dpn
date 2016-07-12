from __future__ import division

import os
import sys
from datetime import datetime
import threading

import tensorflow as tf
from scipy import misc

flags = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_root",
                    os.path.expanduser("~") + "/documents/datasets/voc",
                    "directory contains train and test images")
tf.app.flags.DEFINE_string("output_root", "../data/",
                           "directory to store the formatted data")
tf.app.flags.DEFINE_integer("num_threads", 4,
                            "Number of threads to preprocess the images")


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image, label=None):
    """Convert the images

    Args:
        image: Image to be segmented
        label: Groundtruth for the image segmentation, if `None`, ignore the entry in the example

    Returns:
        example correspond to the origin image label

    """
    if label is None:
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image.tostring())
        }))
    else:
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(label.tostring()),
            'image_raw': _bytes_feature(image.tostring())
        }))
    return example


def _process_image(img):
    """Preprocess the image."""
    return img


def read_files(paths, num_threads):
    """Read file from `files` with mutli threads

    Args:
        paths: absolute path of the files
        num_threads: read file with `num_threads` threads

    Returns:
        A list contains all the file in `files`

    """
    def read_one_thread(paths, list, thread_index):
        print("{}: thread {} start".format(datetime.now(), thread_index))
        for path in paths:
            img = misc.imread(path)
            img = _process_image(img)
            list.append(img)

    coord = tf.train.Coordinator()
    num_per_thread = len(paths) // num_threads
    threads = []
    images_list = [[] for i in range(4)]
    for i in range(num_threads):
        args = (paths[i * num_per_thread:
                      min((i + 1) * num_per_thread, len(paths))], images_list[i], i)
        t = threading.Thread(target=read_one_thread, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    images = reduce(lambda x, y: x + y, images_list, [])
    print("{}: reading {} images complete".format(datetime.now(), len(images)))
    sys.stdout.flush()
    return images


def create_dataset(img_names, img_root, label_root, output_path):
    """Create dataset given images' name.

    Args:
        img_names: A string corresponding to a file which contains image's name
        img_root: Path contains the files in img_names
        img_root: Path contains the labels in img_names, ignored if `None`
        output_path: Path to store the record file

    Returns:
        None

    """
    def save_to_records(save_path, images, labels):
        # images: float32, labels: int32
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in xrange(len(images)):
            if labels:
                example = _convert_to_example(images[i], labels[i])
            else:
                example = _convert_to_example(images[i])
            writer.write(example.SerializeToString())
        print("{}: save to {}".format(datetime.now(), save_path))

    with open(img_names, 'r') as file:
        names = file.readlines()
        names = [name.strip() for name in names]
        images = [os.path.join(img_root, name + ".jpg") for name in names]
        num_threads = flags.num_threads
        images = read_files(images, num_threads)
        labels = None
        if label_root:
            labels = [os.path.join(label_root, name + ".png")
                      for name in names]
            labels = read_files(labels, num_threads)

    save_to_records(output_path, images, labels)


def create_all_dataset(data_root, output_root):
    """Create train, val, test datasets.

    Args:
        data_root: directory contains all the origin data

    Returns: None

    """
    print("{}: creating the dataset".format(datetime.now()))
    image_path = os.path.join(data_root, "JPEGImages")
    label_path = os.path.join(data_root, "SegmentationClass")

    train_file = os.path.join(data_root, "ImageSets",
                              "Segmentation", "train.txt")
    train_output_path = os.path.join(output_root, "train.tf")
    create_dataset(train_file, image_path, label_path, train_output_path)

    valid_file = os.path.join(data_root, "ImageSets",
                              "Segmentation", "val.txt")
    valid_output_path = os.path.join(output_root, "val.tf")
    create_dataset(valid_file, image_path, label_path, valid_output_path)

    test_file = os.path.join(data_root, "ImageSets",
                             "Segmentation", "test.txt")
    test_output_path = os.path.join(output_root, "test.tf")
    create_dataset(test_file, image_path, None, test_output_path)

if __name__ == '__main__':
    create_all_dataset(flags.data_root, flags.output_root)
