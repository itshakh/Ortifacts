import numpy
import matplotlib.image as img

class Artifactory:
    """
    This class helps to generate artificial artifacts to real ortophoto images
    """

    def __init__(self):
        self.image = None
        self.height = None
        self.width = None

    def set_image(self, image):

        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]