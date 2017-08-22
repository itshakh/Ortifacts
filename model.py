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


def batch_norm(in_tensor, phase_train, name, decay=0.99):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
        decay:       decay factor
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name) as scope:
        return tf.contrib.layers.batch_norm(in_tensor, is_training=phase_train, decay=decay, scope=scope)

def batch_norm_wrapper(inputs, is_training, decay = 0.999, epsilon = 1e-3):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0, 1, 2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


# Create model
def net(input, weights, biases, FLAGS, is_training, dropout=0.75):
    # Reshape input picture
    # x = tf.reshape(input, shape=[-1, FLAGS['patch_size'], FLAGS['patch_size'], 3])
    # x = batch_norm(input, is_training, name="x", decay=0.99)
    x = batch_norm(input, is_training, name="input", decay=0.99) # batch_norm_wrapper(input, is_training, decay=0.999, epsilon=1e-3)
    # Convolution Layer

    conv1 = conv2d(x, weights['wc1'])
    reg1 = tf.nn.l2_loss(weights['wc1'])
    # conv1 = batch_norm(conv1, is_training, name="conv1", decay=0.99)
    # conv1 = batch_norm_wrapper(conv1, is_training, decay=0.999, epsilon=1e-3)

    conv1 = tf.nn.relu(conv1)
    conv1 = batch_norm(conv1, is_training, name="conv1", decay=0.99)
    pool1 = maxpool2d(conv1) # 96 * 3 to 48 * 50

    conv2 = conv2d(pool1, weights['wc2'])
    reg2 = tf.nn.l2_loss(weights['wc2'])
    conv2 = tf.nn.relu(conv2)
    conv2 = batch_norm(conv2, is_training, name="conv2", decay=0.99)  # batch_norm_wrapper(conv2, is_training, decay=0.999, epsilon=1e-3)
    pool2 = maxpool2d(conv2) # 44 * 50 to 22 * 100

    conv3 = conv2d(pool2, weights['wc3'])
    # conv3 = batch_norm(conv3, is_training, "conv3", 0.99)
    reg3 = tf.nn.l2_loss(weights['wc3'])
    # Max Pooling (down-sampling)
    conv3 = tf.nn.relu(conv3)
    conv3 = batch_norm(conv3, is_training, name="conv3",
                       decay=0.99)  # batch_norm_wrapper(conv3, is_training, decay=0.999, epsilon=1e-3)
    pool3 = maxpool2d(conv3)# 18 * 100 to 9 * 1

    conv4 = conv2d(pool3, weights['wc4'])
    # conv3 = batch_norm(conv3, is_training, "conv3", 0.99)
    reg4 = tf.nn.l2_loss(weights['wc4'])
    # Max Pooling (down-sampling)
    conv4 = tf.nn.relu(conv4)
    conv4 = batch_norm(conv4, is_training, name="conv4",
                       decay=0.99)  # batch_norm_wrapper(conv3, is_training, decay=0.999, epsilon=1e-3)
    pool4 = maxpool2d(conv4)# 18 * 100 to 9 * 1

    conv5 = conv2d(pool4, weights['wc5'])
    # conv3 = batch_norm(conv3, is_training, "conv3", 0.99)
    reg5 = tf.nn.l2_loss(weights['wc5'])
    # Max Pooling (down-sampling)
    conv5 = tf.nn.relu(conv5)
    conv5 = batch_norm(conv5, is_training, name="conv5",
                       decay=0.99)  # batch_norm_wrapper(conv3, is_training, decay=0.999, epsilon=1e-3)
    pool5 = maxpool2d(conv5)# 18 * 100 to 9 * 1

    # pool3 = tf.nn.dropout(pool3, dropout)
    '''
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
    '''
    # Output, class prediction
    # out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    #pre_fc1 = tf.contrib.layers.flatten(pool5) # tf.reshape(pool5, [-1])

    #fc1 = tf.add(tf.matmul(pre_fc1, weights['wc6']), biases['bc6'])
    #fc1 = tf.nn.sigmoid(fc1)
    # fc1 = tf.nn.dropout(tf.contrib.layers.flatten(pool5), dropout)
    # out = tf.add(tf.matmul(tf.contrib.layers.flatten(pool5), weights['out']), biases['out'])
    out = tf.add(tf.matmul(tf.contrib.layers.flatten(pool5), weights['out']), biases['out'])

    out = tf.nn.sigmoid(out)
    return out, reg1 + reg2 + reg3 + reg4 + reg5
