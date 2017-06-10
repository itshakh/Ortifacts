import tensorflow as tf
import numpy


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def minpool2d(x, k=2):
    # MaxPool2D wrapper
    temp = tf.nn.max_pool(tf.negative(x), ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return tf.negative(temp)


def conv2d(x, W, strides=1):
    # Conv2D wrapper, with NO bias and relu activation
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')


def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon = 1e-3, l_type='fc'):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if l_type == 'conv':
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


# Create model
def net(input, weights, biases, FLAGS, is_training=True, dropout=0.75):
    # Reshape input picture
    x = tf.reshape(input, shape=[-1, FLAGS['patch_size'], FLAGS['patch_size'], 3])
    x = batch_norm_wrapper(x, is_training, l_type='conv')
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'])
    conv1 = batch_norm_wrapper(conv1, is_training, decay=0.999, epsilon=1e-3, l_type='conv')
    conv1 = tf.nn.relu(conv1)

    pool1 = maxpool2d(conv1)

    conv2 = conv2d(pool1, weights['wc2'])
    conv2 = batch_norm_wrapper(conv2, is_training, decay=0.999, epsilon=1e-3, l_type='conv')
    conv2 = tf.nn.relu(conv2)

    pool2 = maxpool2d(conv2)

    conv3 = conv2d(pool2, weights['wc3'])
    conv3 = batch_norm_wrapper(conv3, is_training, decay=0.999, epsilon=1e-3, l_type='conv')
    # Max Pooling (down-sampling)
    max_pool = maxpool2d(conv3, k=FLAGS['patch_size'] - int(weights['wc3'].get_shape().as_list()[0] / 2))
    # Min Pooling (down-sampling)
    min_pool = minpool2d(conv3, k=FLAGS['patch_size'] - int(weights['wc3'].get_shape().as_list()[0] / 2))

    fc1 = tf.concat([tf.reshape(max_pool, [-1, int(weights['wc3'].get_shape().as_list()[-1])]),
                    tf.reshape(min_pool, [-1, int(weights['wc3'].get_shape().as_list()[-1])])], axis=1)

    # Fully connected layer
    fc2 = tf.add(tf.matmul(fc1, weights['wc4']), biases['bc4'])
    fc2 = batch_norm_wrapper(fc2, is_training)
    fc2 = tf.nn.sigmoid(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Fully connected layer 2
    fc3 = tf.add(tf.matmul(fc2, weights['wc5']), biases['bc5'])
    fc3 = batch_norm_wrapper(fc3, is_training)
    fc3 = tf.nn.sigmoid(fc3)

    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    out = tf.nn.sigmoid(out)
    return out
