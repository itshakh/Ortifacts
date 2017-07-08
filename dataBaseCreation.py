import numpy
import matplotlib.image as img
# for debug import matplotlib.pyplot as plt
import Artifactory
import os
import data
import shutil


patch_size = 100
patch_artifact_enlargement = 0.5
max_angle = 15
min_angle = 5
num_of_examps = 1000
# percentages

rotation_prob = 0.1
validation_examps = 0.1
test_examps = 0.1
data_seperation = 0.5
images_path = r"../Raw"

path_to_validation_images = './images/val'
path_to_traning_images = './images/train'
path_to_test_images = './images/test'

articactor = Artifactory.Artifactory()

try:
    shutil.rmtree('./images')
    os.mkdir('./images')
except IOError:
    pass
try:
    os.mkdir(path_to_validation_images)
except IOError:
    pass
try:
    os.mkdir(path_to_traning_images)
except IOError:
    pass
try:
    os.mkdir(path_to_test_images)
except IOError:
    pass
try:
    os.mkdir(os.path.join(path_to_validation_images, '0'))
except IOError:
    pass
try:
    os.mkdir(os.path.join(path_to_validation_images, '1'))
except IOError:
    pass

try:
    os.mkdir(os.path.join(path_to_traning_images, '0'))
except IOError:
    pass
try:
    os.mkdir(os.path.join(path_to_traning_images, '1'))
except IOError:
    print("Could not create folder:: " + os.path.join(path_to_traning_images, '1'))
try:
    os.mkdir(os.path.join(path_to_test_images, '0'))
except IOError:
    print("Could not create folder:: " + os.path.join(path_to_test_images, '0'))
try:
    os.mkdir(os.path.join(path_to_test_images, '1'))
except IOError:
    print("Could not create folder:: " + os.path.join(path_to_test_images, '1'))

rot_90_mat = numpy.array([[0, -1], [1, 0]], dtype=numpy.float32)

images_names = os.listdir(r"../Raw")
data_set = data.Data('p025_10000')
data_set_x = numpy.empty(shape=(num_of_examps, patch_size, patch_size, 3), dtype=numpy.uint8)
data_set_y = numpy.empty(shape=(num_of_examps, 2), dtype=numpy.float32)
numpy.random.seed(42)
for im, image_name in enumerate(images_names):
    input_image = img.imread(os.path.join(images_path, image_name), 'tif')
    for k in range(int(num_of_examps / len(images_names))):
        articactor.set_image(input_image,
                             int(patch_size + patch_size * patch_artifact_enlargement),
                             int(patch_size + patch_size * patch_artifact_enlargement))
        patch_large = articactor.get_random_patch()

        y = numpy.zeros(shape=2, dtype=numpy.float32)
        y[0] = 1.

        # p probability for misalignment
        choise = 0
        if int(num_of_examps / len(images_names)) * im + k < data_seperation * num_of_examps: # numpy.random.choice(2, 1, p=[artifact_prob, 1-artifact_prob])
            choise = 1
        if choise:
            choise2 = numpy.random.choice(2, 1, p=[0.5, 0.5])
            if choise2:
                patch, seam = articactor.patch_misalignment(patch_large, min_angle, max_angle, vertical_seam=True)

                patch = patch[ \
                              int(patch_size *
                              patch_artifact_enlargement/2):-int(patch_size * patch_artifact_enlargement/2),
                              int(patch_size *
                              patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]

                seam = seam[ \
                              int(patch_size *
                              patch_artifact_enlargement/2):-int(patch_size * patch_artifact_enlargement/2),
                              int(patch_size *
                              patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]
                '''
                    plt.subplot(1,2,1)
                    plt.imshow(patch)
                    plt.subplot(1,2,2)
                    plt.imshow(seam*255)
                    plt.show()
                '''

            else:
                patch, seam = articactor.patch_misalignment(patch_large, min_angle, max_angle, vertical_seam=False)

                patch = patch[ \
                              int(patch_size *
                              patch_artifact_enlargement/2):-int(patch_size * patch_artifact_enlargement/2),
                              int(patch_size *
                              patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]

                seam = seam[ \
                              int(patch_size *
                              patch_artifact_enlargement/2):-int(patch_size * patch_artifact_enlargement/2),
                              int(patch_size *
                              patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]
                '''
                    plt.subplot(1,2,1)
                    plt.imshow(patch)
                    plt.subplot(1,2,2)
                    plt.imshow(seam*255)
                    plt.show()
                '''

            y[0] = 0.
            y[1] = 1.
        else:
            patch = patch_large[ \
                              int(patch_size *
                              patch_artifact_enlargement/2):-int(patch_size * patch_artifact_enlargement/2),
                              int(patch_size *
                              patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]

        if not numpy.random.choice(2, 1, p=[rotation_prob, 1-rotation_prob]):
            patch = patch.transpose([1,0,2])
            print("rotated 90 deg")

        data_set_x[int(num_of_examps / len(images_names)) * im + k, :, :, :] = patch
        data_set_y[int(num_of_examps / len(images_names)) * im + k, :] = y

        print('Image ' + image_name + ' patch number {0:d}, choise {1:d}'.format(
            int(num_of_examps / len(images_names)) * im + k, int(choise)))


data_set.x = data_set_x
data_set.y = data_set_y
# shuffle all examples
data_set.shuffle()
# split data to train, validation and test
data_set.set_train_val_batch(int(validation_examps * num_of_examps), int(test_examps * num_of_examps))

# save data to folders

for k in range(data_set.val_x.shape[0]):
    if data_set.val_y[k][0] == 1:
        '''
            NOT damaged images
        '''
        img.imsave(os.path.join(path_to_validation_images, '0', '{:04d}.bmp'.format(k)),
                   data_set.val_x[k].astype(numpy.uint8))
    else:
        img.imsave(os.path.join(path_to_validation_images, '1', '{:04d}.bmp'.format(k)),
                   data_set.val_x[k].astype(numpy.uint8))

for k in range(data_set.train_x.shape[0]):
    if data_set.train_y[k][0] == 1:
        '''
            NOT damaged images
        '''
        img.imsave(os.path.join(path_to_traning_images, '0', '{:04d}.bmp'.format(k)),
                   data_set.train_x[k].astype(numpy.uint8))
    else:
        img.imsave(os.path.join(path_to_traning_images, '1', '{:04d}.bmp'.format(k)),
                   data_set.train_x[k].astype(numpy.uint8))

for k in range(data_set.test_x.shape[0]):
    if data_set.test_y[k][0] == 1:
        '''
            NOT damaged images
        '''
        img.imsave(os.path.join(path_to_test_images, '0', '{:04d}.bmp'.format(k)),
                   data_set.test_x[k].astype(numpy.uint8))
    else:
        img.imsave(os.path.join(path_to_test_images, '1', '{:04d}.bmp'.format(k)),
                   data_set.test_x[k].astype(numpy.uint8))

data_set.export_data(os.path.join(path_to_traning_images, 'train.pickle'),
                     os.path.join(path_to_validation_images, 'val.pickle'),
                     os.path.join(path_to_test_images, 'test.pickle'))
