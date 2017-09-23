import numpy
import matplotlib.image as img
import scipy.ndimage.interpolation as interp
import scipy.ndimage.filters as filter
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
    def get_rot_matrix(angle):
        """
        Generate random rotation matrix
        :param max_angle:
        :return:
        """
        theta = angle * (numpy.pi / 180)

        matrix = numpy.ndarray(shape=(2, 2), dtype=numpy.float32)
        matrix[0, 0] = numpy.cos(theta)
        matrix[0, 1] = -numpy.sin(theta)
        matrix[1, 0] = numpy.sin(theta)
        matrix[1, 1] = matrix[0, 0]

        return matrix

    @staticmethod
    def get_random_rot_matrix(min_angle, max_angle):
        """
        Generate random rotation matrix
        :param max_angle:
        :return:
        """
        theta = numpy.random.uniform(min_angle, max_angle) * (numpy.pi / 180)

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

        return numpy.uint8(numpy.copy(self.image[y:y + self.patch_height, x:x + self.patch_width]))

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

            return numpy.uint8(out)

    @staticmethod
    def merge_patches(patch, sub_patch, seam_mask, feather=False, kernel_size_pix=10):
        sub_patch[sub_patch == 0] = patch[sub_patch == 0]  # fill_holes

        if not feather:
            out = sub_patch  # static left side
            out[seam_mask == 1] = patch[seam_mask == 1]  # distort right side
        else:
            weighted_mask = numpy.zeros_like(seam_mask, dtype=numpy.float32)
            weighted_mask[:, :, 0] = filter.gaussian_filter(input=255*seam_mask[:, :, 0], sigma=kernel_size_pix, truncate=2.0)
            weighted_mask[:, :, 1] = filter.gaussian_filter(input=255*seam_mask[:, :, 1], sigma=kernel_size_pix, truncate=2.0)
            weighted_mask[:, :, 2] = filter.gaussian_filter(input=255*seam_mask[:, :, 2], sigma=kernel_size_pix, truncate=2.0)

            weighted_mask = weighted_mask/255
            out = numpy.zeros(shape=patch.shape, dtype=numpy.float32)
            out += numpy.multiply(weighted_mask, numpy.float32(sub_patch))
            out += numpy.multiply((1.0 - weighted_mask), numpy.float32(patch))
            out = numpy.uint8(out)

        out[out == 0] = 1
        return out

    def patch_misalignment(self, patch, min_rot_angle_deg, max_rot_angle_deg, vertical_seam=True, feather=True):

        seam_mask = self.create_random_seam_mask(patch, vertical_seam)
        seam = self.find_seam(seam_mask)

        matrix = self.get_random_rot_matrix(min_rot_angle_deg, max_rot_angle_deg)
        sub_patch_left = self.rotate_patch(patch, matrix)
        out = self.merge_patches(patch=patch, sub_patch=sub_patch_left, feather=feather, seam_mask=seam_mask)

        return out, seam

    def create_random_seam_mask(self, patch, vertical_seam=True):

        mask_out = numpy.zeros_like(patch, dtype=numpy.float32)
        random_x = numpy.random.randint(patch.shape[0] / 4, 3 * patch.shape[0] / 4)

        if vertical_seam:
            mask_out[:, :random_x, 0:3] = [1, 1, 1]

        else:
            for row in range(0, self.patch_height):
                mask_out[row, :random_x, 0:3] = [1, 1, 1]
                random_x += numpy.random.randint(0, 3) - 1
                random_x = min(random_x, mask_out.shape[1] - 1)
                random_x = max(random_x, 0)

        return numpy.float32(mask_out)

    @staticmethod
    def find_seam(seam_mask):

        shifted_seam = numpy.zeros_like(seam_mask, dtype=numpy.float32) + 1
        shifted_seam[:, 1:, 0:3] = seam_mask[:, :-1, 0:3]
        seam = numpy.absolute(seam_mask - shifted_seam)

        return seam.astype(numpy.float32)



