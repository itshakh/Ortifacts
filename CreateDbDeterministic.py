import numpy
import random as random
import matplotlib.image as img
import Artifactory
import os
import shutil


# properties
patch_size = 224
patch_artifact_enlargement = 0.5
max_angle = 20
min_angle = 10

# data set size
data_base_size_unit = 1000
train_size_factor = 1.0
validation_size_factor = 0.1
test_size_factor = 0.1

# paths
images_path = r"../Raw"
path_to_train_images = './images/train'
path_to_val_images = './images/val'
path_to_test_images = './images/test'

factory = Artifactory.Artifactory()

# delete existing images
try:
    shutil.rmtree('./images')
    os.mkdir('./images')
except IOError:
    raise IOError('can not delete existing images')

# create db folders
try:
    os.mkdir(path_to_train_images)
    os.mkdir(os.path.join(path_to_train_images, '0'))
    os.mkdir(os.path.join(path_to_train_images, '1'))

    os.mkdir(path_to_val_images)
    os.mkdir(os.path.join(path_to_val_images, '0'))
    os.mkdir(os.path.join(path_to_val_images, '1'))

    os.mkdir(path_to_test_images)
    os.mkdir(os.path.join(path_to_test_images, '0'))
    os.mkdir(os.path.join(path_to_test_images, '1'))

except IOError:
    raise IOError('cannot create db folders')

random.seed(42)

# patches arithmetic
images_names = os.listdir(r"../Raw")
num_images = len(images_names)
total_num_of_patches = data_base_size_unit * (train_size_factor + test_size_factor + validation_size_factor)
patches_per_image = int(numpy.ceil(total_num_of_patches / num_images))

num_of_val_patches_per_image = int(patches_per_image * validation_size_factor)
num_of_test_patches_per_image = int(patches_per_image * test_size_factor)
num_of_train_patches_per_image = int(patches_per_image - num_of_val_patches_per_image - num_of_test_patches_per_image)

if num_of_train_patches_per_image % 2 != 0:
    num_of_train_patches_per_image -= 1
    patches_per_image -= 1

# prints
print('|-------------------------------------|')
print('|          DATA BASE STATS:           |')
print('|-------------------------------------|')
print('|{:10} | {:5}|         Global|' .format('  Src  Images', num_images))
print('|{:10} | {:5}|         Global|' .format('Patch    Size', patch_size))
print('|{:10} | {:5}|         Global|' .format('  Min   Angle', min_angle))
print('|{:10} | {:5}|      Per Image|'.format('  Max   Angle', max_angle))
print('|{:10} | {:5}|      Per Image|'.format('Train Patches', num_of_train_patches_per_image))
print('|{:10} | {:5}|      Per Image|'.format('  Val Patches', num_of_val_patches_per_image))
print('|{:10} | {:5}|      Per Image|'.format(' Test Patches', num_of_test_patches_per_image))
print('|-------------------------------------|\n')

# create patches
enlarged_patch_size = int(patch_size * (1 + patch_artifact_enlargement))

# global ids -
train_global_id = 0
val_global_id = 0
test_global_id = 0
global_id = 0

for image, image_name in enumerate(images_names):
    input_image = img.imread(os.path.join(images_path, image_name), 'tif')
    factory.set_image(input_image, enlarged_patch_size, enlarged_patch_size)

    # patch role division
    train_val_test_division = numpy.ones(patches_per_image) * 2
    train_val_test_division[0:num_of_train_patches_per_image] = 0
    train_val_test_division[num_of_train_patches_per_image:-num_of_test_patches_per_image] = 1
    random.shuffle(train_val_test_division)
    
    # patch class division
    train_classes = numpy.zeros(num_of_train_patches_per_image, dtype=numpy.int)
    val_classes = numpy.zeros(num_of_val_patches_per_image, dtype=numpy.int)
    test_classes = numpy.zeros(num_of_test_patches_per_image, dtype=numpy.int)

    train_classes[0: int(num_of_train_patches_per_image / 2)] = 1
    val_classes[0: int(num_of_val_patches_per_image / 2)] = 1
    test_classes[0: int(num_of_test_patches_per_image / 2)] = 1

    random.shuffle(train_classes)
    random.shuffle(val_classes)
    random.shuffle(test_classes)

    image_patch_idx = 0
    for patch_order in range(0, patches_per_image):

        patch_large = factory.get_random_patch()
        cur_role = train_val_test_division[image_patch_idx]
        flip = random.choice([True, False])

        # train:
        if cur_role == 0:
            cur_class = train_classes[train_global_id % num_of_train_patches_per_image]
        # val:
        elif cur_role == 1:
            cur_class = val_classes[val_global_id % num_of_val_patches_per_image]
        # test:
        elif cur_role == 2:
            cur_class = test_classes[test_global_id % num_of_test_patches_per_image]

        if cur_class == 1:
            patch_large, seam = factory.patch_misalignment(patch_large, min_angle, max_angle, vertical_seam=random.choice([True, False]))

        patch = patch_large[
                int(patch_size *
                    patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2),
                int(patch_size *
                    patch_artifact_enlargement / 2):-int(patch_size * patch_artifact_enlargement / 2), :]

        if flip == 1:
            patch = patch.transpose([1, 0, 2])

        # save
        # train:
        if cur_role == 0:
            cur_save_path = os.path.join(path_to_train_images, str(cur_class), '{:05d}.bmp'.format(train_global_id))
        # val:
        elif cur_role == 1:
            cur_save_path = os.path.join(path_to_val_images, str(cur_class), '{:05d}.bmp'.format(val_global_id))
        # test:
        elif cur_role == 2:
            cur_save_path = os.path.join(path_to_test_images, '{}'.format(cur_class), '{:05d}.bmp'.format(test_global_id))

        img.imsave(cur_save_path, patch)
        print('\rPROGRESS ...... {:3d}% done'.format(int(100 * global_id / (patches_per_image * num_images))),
              end='', flush=True)

        train_global_id += 1
        val_global_id += 1
        test_global_id += 1
        global_id += 1
        image_patch_idx += 1

print('\rPROGRESS ...... DONE!', end='', flush=True)
