import tensorflow as tf
import model
import data
import numpy
import matplotlib.pyplot as plt
import Artifactory
import matplotlib.image as img


FLAGS = dict()
FLAGS['patch_size'] = 100
FLAGS['kernel_size'] = 5
FLAGS['num_of_filters1'] = 50
FLAGS['num_of_filters2'] = 100
FLAGS['num_of_filters3'] = 50
FLAGS['input_image_path'] = r'C:\PlayGround\Ortifacts\images\test\1\0028.bmp'
FLAGS['pre_trained_model_path'] = './10_06_17__1.model.ckpt'

# load data
input_image_artifactor = Artifactory.Artifactory()
#input_image_artifactor.set_image(img.imread(FLAGS['input_image_path'], 'bmp'), FLAGS['patch_size'], FLAGS['patch_size'])
input_image = img.imread(FLAGS['input_image_path'], 'bmp')

plt.imshow(input_image)
plt.show()
"""  Create all placeholders of the Net  """
tf.reset_default_graph()
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, FLAGS['patch_size'], FLAGS['patch_size'], 3])
y = tf.placeholder(tf.float32, shape=[None, 2])
training = tf.placeholder(tf.bool, name='training')
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], 3, FLAGS['num_of_filters1']]
        , 0, 1.)),
    'wc2': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters1'], FLAGS['num_of_filters2']]
        , 0, 1.)),
    'wc3': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters2'], 1]
        , 0, 1.)),

    # 'wc4': tf.Variable(tf.random_normal([FLAGS['num_of_filters3'] * 2, 200], 0, 1.)),
    # 'wc5': tf.Variable(tf.random_normal([200, 200], 0, 1.)),

    'out': tf.Variable(tf.random_normal([81, 2], 0, 1.))
}
#variable_summaries(weights)
biases = {
    'bc4': tf.Variable(tf.random_normal([81])),
    'bc5': tf.Variable(tf.random_normal([81])),
    'out': tf.Variable(tf.random_normal([2]))
}

# initialize the CNN
pred = model.net(x, weights, biases, FLAGS, training, keep_prob)
# test_pred = model.net(x, weights, biases, FLAGS, training, keep_prob)

# Performance parameters
tp = tf.count_nonzero(tf.argmax(pred, 1) * tf.argmax(y, 1))
tn = tf.count_nonzero((tf.argmax(pred, 1) - 1) * (tf.argmax(y, 1) - 1))
fp = tf.count_nonzero(tf.argmax(pred, 1) * (tf.argmax(y, 1) - 1))
fn = tf.count_nonzero((tf.argmax(pred, 1) - 1) * tf.argmax(y, 1))

tpr = tp / tf.count_nonzero(tf.argmax(y, 1))
fpr = fp / tf.count_nonzero(1 - tf.argmax(y, 1))
fnr = fn / tf.count_nonzero(1 - tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

# Performance parameters for test phase

prob_out = tf.multiply(1 - pred[:, 0], pred[:, 1])

Positives = tf.argmax(y, 1)

prob_out_positives = tf.multiply(tf.cast(prob_out, tf.float32), tf.cast(Positives, tf.float32))
number_of_high_probs = tf.cast(tf.count_nonzero(prob_out_positives), tf.float32)
values_01, indices_01 = tf.nn.top_k(prob_out_positives, k=tf.cast(tf.ceil(tf.cast(tf.count_nonzero(prob_out_positives), tf.float32) * 0.001), tf.int32))
threshold_99_9 = tf.reduce_min(values_01)

values_0, indices_0 = tf.nn.top_k(prob_out_positives, k=tf.cast(tf.ceil(tf.cast(tf.count_nonzero(prob_out_positives), tf.float32) * 0.01), tf.int32))
threshold_99_0 = tf.reduce_min(values_0)

Negatives = 1 - tf.argmax(y, 1)

False_detection_99_9 = tf.count_nonzero(tf.cast(tf.logical_and(tf.greater(Negatives, 0), tf.greater(prob_out, threshold_99_9)), tf.float32)) / \
                       tf.count_nonzero(Negatives)
False_detection_99_0 = tf.count_nonzero(tf.cast(tf.logical_and(tf.greater(Negatives, 0), tf.greater(prob_out, threshold_99_0)), tf.float32)) / \
                       tf.count_nonzero(Negatives)

# test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_pred, 1), tf.argmax(y, 1)), tf.float32))
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.abs(tf.add(y, tf.negative(pred)))) # L1

losses = []

# Strat session for training
print("Start a session")
sess = tf.Session()
print("Open a saver")
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)
try:
    print("Try to load pre trained model")
    saver.restore(sess, FLAGS['pre_trained_model_path'])
    print("Load previous model")
except:
    print("Could not load model, use model initialization")
    init = tf.global_variables_initializer()
    sess.run(init)

prediction = sess.run([pred], feed_dict={
        x: input_image[:, :, :3].reshape((1, 100, 100,3)), keep_prob: 1., training: False
    })

print("Predicted %d " % numpy.argmax(prediction) + "for label 1")
