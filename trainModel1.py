import tensorflow as tf
import model
import data
import numpy
import matplotlib.pyplot as plt


FLAGS = dict()
FLAGS['patch_size'] = 100
FLAGS['learning_rate'] = 0.001
FLAGS['kernel_size'] = 5
FLAGS['num_of_filters1'] = 50
FLAGS['num_of_filters2'] = 100
FLAGS['num_of_filters3'] = 50
FLAGS['batch_size'] = 20
FLAGS['max_iters'] = 10000000
FLAGS['epochs'] = 50
FLAGS['training'] = True
FLAGS['dropout'] = 0.75
FLAGS['display_step'] = 50
FLAGS['pre_trained_model_path'] = './10_06_17__.model.ckpt'
FLAGS['output_model_path'] = './10_06_17__2.model.ckpt'
FLAGS['train_data_path'] = './images/train/train.pickle'
FLAGS['val_data_path'] = './images/val/val.pickle'
FLAGS['test_data_path'] = './images/test/test.pickle'

# load data
full_data = data.Data('')
full_data.import_data(FLAGS['train_data_path'], FLAGS['val_data_path'], FLAGS['test_data_path'])

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

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate']).minimize(cost)
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

losses_training = []
losses_validation = []
prediction = []

epoch = 0
glob_step = 0
step = 0


if FLAGS['training']:
    while epoch < FLAGS['epochs'] and step < FLAGS['max_iters']:
        full_data.shuffle_train()
        step = 0
        while step * (FLAGS['batch_size'] + 1) < full_data.train_x.shape[0]:
            batch_x, batch_y = full_data.get_train_batch(range(step * FLAGS['batch_size'], (step + 1) * FLAGS['batch_size']))
            # Run optimization op (backprop)
            sess.run([optimizer], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: FLAGS['dropout'], training: True})

            step += 1
            glob_step += 1

            if step % FLAGS['display_step'] == 0:

                train_loss, acc_train, tpr_train, fpr_train,fnr_train = sess.run([cost, accuracy, tpr, fpr, fnr],
                                                                       feed_dict={x: batch_x, y: batch_y, keep_prob: 1.
                                                                                  , training: True})
                val_loss, acc_val, tpr_val, fpr_val, fnr_val, fd_train_99_0, fd_train_99_9 = sess.run([
                    cost, accuracy, tpr, fpr, fnr, False_detection_99_0, False_detection_99_9], feed_dict={
                    x: full_data.val_x[:250], y: full_data.val_y[:250], keep_prob: 1., training: False
                })
                # train_writer.add_summary(summary, step)
                # prediction.append(prediction_)

                losses_validation.append(val_loss)
                losses_training.append(train_loss)


                print("Epoch " + str(epoch) + ", Iter " + str(step*FLAGS['batch_size'])
                      + "\nTraining Loss = {:.6f}".format(train_loss) +
                      " \nAccuracy = {:.6f}".format(acc_train)+
                      " \nFPR = {:.6f}".format(fpr_train) +
                      " \nTPR = {:.6f}".format(tpr_train)+
                      " \nFNR = {:.6f}".format(fnr_train))

                print("\nValidation Loss = " + "{:.6f}".format(val_loss) +
                      " \nAccuracy = {:.6f}".format(acc_val) +
                      " \nFPR = {:.6f}".format(fpr_val) +
                      " \nTPR = {:.6f}".format(tpr_val) +
                      " \nFNR = {:.6f}".format(fnr_val) +
                      " \nFalse Detection 99.9 = {:6f}".format(fd_train_99_9) +
                      " \nFalse Detection 99 = {:6f}".format(fd_train_99_0))

                if tpr_val > 0.97 and fpr_val < 0.03:
                    break
                # save_path = saver.save(sess, r"E:\studies\NetworkSeg\checkpoints\model.ckpt")
                # print("Model saved in file: %s" % save_path)
        epoch += 1
if FLAGS['training']:
    save_path = saver.save(sess, FLAGS['output_model_path'])
    print("Final model saved in file: %s" % save_path)
    # sess.close()

test_acc = sess.run(test_accuracy, feed_dict={x: full_data.test_x[:200], y: full_data.test_y[:200], keep_prob: 1.,
                                              training: False})
train_acc = sess.run(test_accuracy, feed_dict={x: full_data.train_x[:200], y: full_data.train_y[:200], keep_prob: 1.,
                                               training: False})
val_acc = sess.run(test_accuracy, feed_dict={x: full_data.val_x[:200], y: full_data.val_y[:200], keep_prob: 1.,
                                             training: False})

print("train accuracy = {:f} val accuracy = {:f} and test accuracy = {:f}".format(train_acc, val_acc, test_acc))

