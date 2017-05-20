import numpy
import matplotlib.image as img
import matplotlib.pyplot as plt
import Artifactory


patch_size = 100
max_angle = 10

input_image = img.imread(r"C:\PlayGround\Raw\42825.tif", 'tif')

articactor = Artifactory.Artifactory()

articactor.set_image(input_image, patch_size, patch_size)

patch = articactor.get_random_patch()

misaligned_patch = articactor.patch_misalignment(patch, max_angle)
plt.imshow(misaligned_patch)
plt.show()
a = 1