from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy
import matplotlib.image as im
import matplotlib.pyplot as plt
import os
import scipy
import pickle
import random

def encode_label(label):
  return int(label)

def read_label_file(file):
  f = open(file, "r")
  filepaths = []
  labels = []
  for line in f:
    filepath, label = line.split(",")
    filepaths.append(filepath)
    labels.append(encode_label(label))
  return filepaths, labels

class Data:
    def __init__(self, name):
        """
        :train :a tuple of x and y (data and labels)
        :test :a tuple of x and y (data and labels)
        :return:
        """
        self.x = None
        self.y = None

        self.val_x = None
        self.val_y = None

        self.test_x = None
        self.test_y = None

        self.train_x = None
        self.train_y = None

        self.data_type = 'csv'
        self.name = name

        self.train_images = None
        self.train_labels = None
        self.val_images = None
        self.val_labels = None
        self.test_images = None
        self.test_labels = None

        self.train_input_queue = None
        self.val_input_queue = None
        self.test_input_queue = None

        self.train_image = None
        self.val_image = None
        self.test_image = None
        self.train_label = None
        self.val_label = None
        self.test_label = None
        self.train_image_batch = None
        self.train_label_batch = None
        self.val_image_batch = None
        self.val_label_batch = None
        self.test_image_batch = None
        self.test_label_batch = None

        self.num_of_train_samples = 0
        self.num_of_val_samples = 0
        self.num_of_test_samples = 0

    def shuffle(self):
        if self.data_type != 'csv':
            range_list = list(range(self.x.shape[0]))

            random.shuffle(range_list)

            self.x[list(range(self.x.shape[0]))] = self.x[range_list]
            self.y[list(range(self.y.shape[0]))] = self.y[range_list]

    def shuffle_train(self):
        if self.data_type != 'csv':
            range_list = list(range(self.train_x.shape[0]))

            random.shuffle(range_list)

            self.train_x[list(range(self.train_x.shape[0]))] = self.train_x[range_list]
            self.train_y[list(range(self.train_y.shape[0]))] = self.train_y[range_list]

    def get_batch(self, set_range):
        """
        returns a mini batch of data
        :param kind: 'train', 'test' or 'validation'
        :param set_range:
        :return:
        """

        return self.x[set_range], self.y[set_range]

    def get_train_batch(self, set_range):
        """
        Return a batch for training
        :param set_range:
        :return:
        """
        return self.train_x[set_range], self.train_y[set_range]

    def set_train_val_batch(self, val_amount, test_amount):
        """

        :param val_amount: number of validation images
        :param test_amount: -"- test -"-
        :return:
        """

        self.train_x = self.x[:self.x.shape[0] - val_amount - test_amount]
        self.train_y = self.y[:self.x.shape[0] - val_amount - test_amount]

        self.val_x = self.x[self.train_x.shape[0]: self.train_x.shape[0] + val_amount]
        self.val_y = self.y[self.train_x.shape[0]: self.train_x.shape[0] + val_amount]

        self.test_x = self.x[self.x.shape[0] - test_amount:self.x.shape[0]]
        self.test_y = self.y[self.x.shape[0] - test_amount:self.x.shape[0]]

    def export_data(self, train_path, val_path, test_path):
        """
            save data as pickle
        """
        pickle.dump((self.train_x, self.train_y), open(train_path, "wb"))
        pickle.dump((self.val_x, self.val_y), open(val_path, "wb"))
        pickle.dump((self.test_x, self.test_y), open(test_path, "wb"))

    def import_data(self, train_path, val_path, test_path, image_width=100, image_height=100, num_of_channels=3):
        """
            load pickles
        """

        if self.data_type == 'csv':
            # reading labels and file path
            train_filepaths, train_labels = read_label_file(train_path)
            val_filepaths, val_labels = read_label_file(val_path)
            test_filepaths, test_labels = read_label_file(test_path)

            self.num_of_train_samples = len(train_filepaths)
            self.num_of_val_samples = len(val_filepaths)
            self.num_of_test_samples = len(test_filepaths)

            # convert string into tensors
            self.train_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
            self.train_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)
            self.val_images = ops.convert_to_tensor(val_filepaths, dtype=dtypes.string)
            self.val_labels = ops.convert_to_tensor(val_labels, dtype=dtypes.int32)
            self.test_images = ops.convert_to_tensor(test_filepaths, dtype=dtypes.string)
            self.test_labels = ops.convert_to_tensor(test_labels, dtype=dtypes.int32)

            # create input queues
            self.train_input_queue = tf.train.slice_input_producer(
                [self.train_images, self.train_labels],
                shuffle=True)

            self.val_input_queue = tf.train.slice_input_producer(
                [self.val_images, self.val_labels],
                shuffle=True)

            self.test_input_queue = tf.train.slice_input_producer(
                [self.test_images, self.test_labels],
                shuffle=True)

            # process path and string tensor into an image and a label
            train_file_content = tf.read_file(self.train_input_queue[0])
            self.train_image = tf.image.decode_png(train_file_content, channels=num_of_channels)
            self.train_label = self.train_input_queue[1]

            val_file_content = tf.read_file(self.val_input_queue[0])
            self.val_image = tf.image.decode_png(val_file_content, channels=num_of_channels)
            self.val_label = self.val_input_queue[1]

            test_file_content = tf.read_file(self.test_input_queue[0])
            self.test_image = tf.image.decode_png(test_file_content, channels=num_of_channels)
            self.test_label = self.test_input_queue[1]

            # define tensor shape
            self.train_image.set_shape([image_height, image_width, num_of_channels])
            self.val_image.set_shape([image_height, image_width, num_of_channels])
            self.test_image.set_shape([image_height, image_width, num_of_channels])

        else:
            (self.train_x, self.train_y) = pickle.load(open(train_path, "rb"))
            (self.val_x, self.val_y) = pickle.load(open(val_path, "rb"))
            (self.test_x, self.test_y) = pickle.load(open(test_path, "rb"))

    def set_batches(self, train_batch_size, test_batch_size, val_batch_size):
        # collect batches of images before processing
        self.train_image_batch, self.train_label_batch = tf.train.batch(
            [self.train_image, self.train_label],
            batch_size=train_batch_size
            # ,num_threads=1
        )
        self.val_image_batch, self.val_label_batch = tf.train.batch(
            [self.val_image, self.val_label],
            batch_size=val_batch_size
            # ,num_threads=1
        )

        self.test_image_batch, self.test_label_batch = tf.train.batch(
            [self.test_image, self.test_label],
            batch_size=test_batch_size
            # ,num_threads=1
        )
        # return train_image_batch, train_label_batch


    @staticmethod
    def feather(mask, image1, image2, ksize=3):
        kernel = numpy

