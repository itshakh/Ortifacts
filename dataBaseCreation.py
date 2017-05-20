import numpy
import matplotlib.image as img
import matplotlib.pyplot as plt
import Artifactory
import os
import data


patch_size = 100
max_angle = 10
num_of_examps = 10000
artifact_prob = 0.25
images_path = r"../Raw"

articactor = Artifactory.Artifactory()

# articactor.set_image(input_image, patch_size, patch_size)


images_names = os.listdir(r"../Raw")
data_set = data.Data('p025_10000')
data_set_x = numpy.empty(shape=(num_of_examps, patch_size, patch_size, 3), dtype=numpy.float32)
data_set_y = numpy.empty(shape=(num_of_examps, 1), dtype=numpy.float32)

for im, image_name in enumerate(images_names):
    input_image = img.imread(os.path.join(images_path, image_name), 'tif')
    for k in range(int(num_of_examps /len(images_names))):
        articactor.set_image(input_image, patch_size, patch_size)
        patch = articactor.get_random_patch()
        y = 0
        # p probability for misalignment
        choise = numpy.random.choice(2, 1, p=[artifact_prob, 1-artifact_prob])
        if choise:
            patch = articactor.patch_misalignment(patch, max_angle)
            y = 1

        data_set_x[im*int((num_of_examps /len(images_names))) + k, :, :, :] = patch
        data_set_y[im*int((num_of_examps /len(images_names))) + k] = y
        print('Image ' + image_name + ' patch number {0:d}, choise {1:d}'.format(
            im*int((num_of_examps /len(images_names))) + k, int(choise)))
data_set.x = data_set_x
data_set.y = data_set_y

data_set.export_data('p025_10000.pickle')