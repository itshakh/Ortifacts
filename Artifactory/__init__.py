import numpy
import matplotlib.image as img
import scipy.ndimage.interpolation as interp
import matplotlib.pyplot as plt


class Artifactory:
    """
    This class helps to generate artificial artifacts to real ortophoto images.
    Optional artifacts:
        - misalignment in the region of stiching
        - Ghosting (FUTURE)
        - TBD
    """

    def __init__(self):

        self.image = None
        self.height = None
        self.width = None
        self.patch_height = None
        self.patch_width = None

    def set_image(self, image, patch_height, patch_width):
        """

        :param image: full size image (rgb)
        :param patch_height:
        :param patch_width:
        :return:
        """
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.patch_height = patch_height
        self.patch_width = patch_width

    @staticmethod
    def get_random_rot_matrix(max_angle):
        """
        Generate random rotation matrix
        :param max_angle:
        :return:
        """
        theta = numpy.random.uniform(0, max_angle) * (numpy.pi / 180)

        matrix = numpy.ndarray(shape=(2, 2), dtype=numpy.float32)
        matrix[0, 0] = numpy.cos(theta)
        matrix[0, 1] = -numpy.sin(theta)
        matrix[1, 0] = numpy.sin(theta)
        matrix[1, 1] = matrix[0, 0]

        return matrix

    def get_random_patch(self):
        """
        Select a random patch out of the image
        :return:
        """
        y = numpy.random.randint(0, self.height - self.patch_height)
        x = numpy.random.randint(0, self.width - self.patch_width)

        return numpy.copy(self.image[y:y + self.patch_height, x:x + self.patch_width])

    @staticmethod
    def rotate_patch(patch, rot_matrix):
        """

        :param patch:
        :param rot_matrix:
        :return:
        """
        if len(patch.shape) != 3:
            # Gray
            return numpy.copy(interp.affine_transform(patch, rot_matrix, offset=(0, 0)))
        else:
            # RGB image
            out = numpy.ndarray(shape=patch.shape, dtype=numpy.float32)
            out[:, :, 0] = interp.affine_transform(patch[:, :, 0], rot_matrix, offset=(0, 0))
            out[:, :, 1] = interp.affine_transform(patch[:, :, 1], rot_matrix, offset=(0, 0))
            out[:, :, 2] = interp.affine_transform(patch[:, :, 2], rot_matrix, offset=(0, 0))

            return out

    def patch_misalignment(self, patch, max_rot_angle):
        """

        :param patch: 2D numpy array (sub image)
        :param max_rot_angle:
        :return: Misaligned image
        """
        random_x_split = numpy.random.randint(0, self.patch_width)

        # artificially rotate sub patch
        matrix = self.get_random_rot_matrix(max_rot_angle)
        sub_patch_left = self.rotate_patch(patch, matrix)

        # merge two sides of the patch into a new 2D image
        # out = numpy.zeros(shape=patch.shape, dtype=numpy.float32)
        out = sub_patch_left
        out[:, random_x_split:] = patch[:, random_x_split:]

        out[out == 0] = patch[out == 0]
        return out
