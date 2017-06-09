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
        self.image[self.image == 0] = 1  # zero values are transparent
		
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

    def create_random_seam_mask(self, patch, vertical_seam=True):

        mask_out = numpy.zeros_like(patch, dtype=numpy.float32)
        random_x = numpy.random.randint(0, patch.shape[0])

        if vertical_seam:
            mask_out[:, :random_x, 0:3] = [255, 255, 255]

        else:
            for row in range(0, self.patch_height):
                mask_out[row, :random_x, 0:3] = [255, 255, 255]
                random_x += numpy.random.randint(0, 3) - 1
                random_x = min(random_x, mask_out.shape[1] - 1)
                random_x = max(random_x, 0)

        return mask_out

    @staticmethod
    def find_seam(seam_mask):

        shifted_seam = numpy.zeros_like(seam_mask, dtype=numpy.float32) + 255
        shifted_seam[:, 1:, 0:3] = seam_mask[:, :-1, 0:3]
        seam = numpy.absolute(seam_mask - shifted_seam)

        return seam.astype(numpy.float32)

