import numpy
import matplotlib.image as im
import matplotlib.pyplot as plt
import os
import scipy
import pickle
import random

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

        self.name = name

    def shuffle(self):
        range_list = list(range(self.x.shape[0]))

        random.shuffle(range_list)

        self.x[list(range(self.x.shape[0]))] = self.x[range_list]
        self.y[list(range(self.y.shape[0]))] = self.y[range_list]

    def shuffle_train(self):
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

    def import_data(self, train_path, val_path, test_path):
        """
            load pickles
        """
        (self.train_x, self.train_y) = pickle.load(open(train_path, "rb"))
        (self.val_x, self.val_y) = pickle.load(open(val_path, "rb"))
        (self.test_x, self.test_y) = pickle.load(open(test_path, "rb"))

    @staticmethod
    def feather(mask, image1, image2, ksize=3):
        kernel = numpy

