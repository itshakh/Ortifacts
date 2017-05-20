import numpy
import matplotlib.image as im
import matplotlib.pyplot as plt
import os
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
        return (self.x[set_range], self.y[set_range])

    def get_train_batch(self, set_range):
        """
        returns a mini batch of data
        :param kind: 'train', 'test' or 'validation'
        :param set_range:
        :return:
        """
        return (self.train_x[set_range], self.train_y[set_range])

    def get_val_batch(self, set_range):
        """
        returns a mini batch of data
        :param kind: 'train', 'test' or 'validation'
        :param set_range:
        :return:
        """

        self.train_x = numpy.ndarray(shape=(self.x.shape[0] - len(list(set_range)), self.x.shape[1], self.x.shape[2], 3),dtype=numpy.float32)
        self.train_y = numpy.ndarray(shape=(self.x.shape[0] - len(list(set_range)), 1), dtype=numpy.float32)
        self.train_x[:set_range[0]] = self.x[:set_range[0]]
        self.train_x[set_range[0] + 1:] = self.x[set_range[-1]+1:]

        self.train_y[:set_range[0]] = self.y[:set_range[0]]
        self.train_y[set_range[0] + 1:] = self.y[set_range[-1]+1:]

        self.val_x = self.x[set_range]
        self.val_y = self.y[set_range]

        return (self.val_x, self.val_y)


    def export_data(self, path):
        """
            save data as pickle
        """
        pickle.dump((self.x, self.y), open( path, "wb" ) )

    def import_data(self, path):
        """
            save data as pickle
        """
        (self.x, self.y) = pickle.load(open( path, "rb" ) )


    def load_data(self, path_to_images, names_file_path, scores_file_path, patch_size=(32, 32)):
        f1 = open(names_file_path)
        names = f1.readlines()
        f2 = open(scores_file_path)
        scores = f2.readlines()
        x = numpy.ndarray(shape=(len(names), patch_size[0], patch_size[1], 3),dtype=numpy.float32)
        y = numpy.ndarray(shape=(len(names), 1), dtype=numpy.float32)

        full_data = (x, y)
        current_example = 0
        for name, score, k in zip(names, scores, range(len(names))):
            print("Load image number {:d} / {:d}".format(k, len(names)))
            type = name.split('.')[-1].strip('\n')
            # temp_image = functions.contrast_normalize(im.imread(os.path.join(path_to_images, name.strip('\n')), type))
            temp_image = numpy.float32(im.imread(os.path.join(path_to_images, name.strip('\n')), type))
            temp_score = numpy.float32(score)
            slices = (int(temp_image.shape[0] / patch_size[0]), int(temp_image.shape[1] / patch_size[1]))

            if x.shape[0] - current_example <= slices[0] * slices[1]:
                temp = numpy.ndarray(shape=(x.shape[0] + (slices[0] * slices[1] - x.shape[0] + current_example),
                                            patch_size[0], patch_size[1], 3), dtype=numpy.float32)
                temp[:x.shape[0]] = x

                x = temp

                temp = numpy.ndarray(shape=(y.shape[0] + (slices[0] * slices[1] - y.shape[0] + current_example), 1),
                                     dtype=numpy.float32)
                temp[:y.shape[0]] = y

                y = temp

            for slice_x in range(slices[0]):
                for slice_y in range(slices[1]):

                    x[current_example] = \
                        temp_image[
                        slice_x * patch_size[0]:(slice_x + 1) * patch_size[0],
                        slice_y * patch_size[1]:(slice_y + 1) * patch_size[1]
                        ]
                    y[current_example] = temp_score / 100
                    current_example += 1
            self.x = x
            self.y = y
