import tensorflow as tf
import model
import data
import numpy
import sys
import matplotlib.pyplot as plt


FLAGS = dict()

# Network architecture
FLAGS['patch_size'] = 100
FLAGS['learning_rate'] = 0.0001
FLAGS['kernel_size'] = 3
FLAGS['num_of_filters1'] = 32
FLAGS['num_of_filters2'] = 64
FLAGS['num_of_filters3'] = 64
FLAGS['num_of_filters4'] = 64
FLAGS['num_of_filters5'] = 64

# training parameters
FLAGS['step_size'] = 10000
FLAGS['etha'] = 0.1

FLAGS['batch_size'] = 20
FLAGS['max_iters'] = 10000000
FLAGS['epochs'] = 40
FLAGS['training'] = True
FLAGS['dropout'] = 0.9
FLAGS['display_step'] = 50
FLAGS['Beta'] = 0.01

# output and input paths
FLAGS['pre_trained_model_path'] = './models/single_output_3.model.ckpt' #'./models/07_07_17__6.model.ckpt'
FLAGS['output_model_path'] = './models/single_output_4.model.ckpt'
FLAGS['train_data_path'] = './images/train/train-labels.csv'
FLAGS['val_data_path'] = './images/val/val-labels.csv'
FLAGS['test_data_path'] = './images/test/test-labels.csv'


"""  Create all placeholders of the Net  """
tf.reset_default_graph()

# load data
full_data = data.Data('')
full_data.import_data(FLAGS['train_data_path'], FLAGS['val_data_path'], FLAGS['test_data_path'],
                      FLAGS['patch_size'], FLAGS['patch_size'], num_of_channels=3)


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, FLAGS['patch_size'], FLAGS['patch_size'], 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
training = tf.placeholder(tf.bool, name='training')

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 16 outputs
    'wc1': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], 3, FLAGS['num_of_filters1']]
        , 0, 0.01)),
    'wc2': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters1'], FLAGS['num_of_filters2']]
        , 0, 0.01)),
    'wc3': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters2'], FLAGS['num_of_filters3']]
        , 0, 0.01)),

    'wc4': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters3'], FLAGS['num_of_filters4']]
        , 0, 0.01)),

    'wc5': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters4'], FLAGS['num_of_filters5']]
        , 0, 0.01)),

    'out': tf.Variable(tf.random_normal([256, 1], 0, 0.01))
}

biases = {
    'out': tf.Variable(tf.zeros((1)))
}
# initialize the CNN
pred, reg, feature_maps_summary = model.net(x, weights, biases, FLAGS, training, keep_prob)

# test_pred = model.net(x, weights, biases, FLAGS, training, keep_prob)

# Performance parameters
tp = tf.count_nonzero(tf.cast(tf.greater(pred, 0.5), tf.float32) * tf.cast(tf.greater(y, 0.5), tf.float32)) # tf.count_nonzero(tf.argmax(pred, 1) * tf.argmax(y, 1))
tn = tf.count_nonzero((tf.cast(tf.greater(pred, 0.5), tf.float32) - 1) * (tf.cast(tf.greater(y, 0.5), tf.float32) - 1))# tf.count_nonzero((tf.argmax(pred, 1) - 1) * (tf.argmax(y, 1) - 1))
fp = tf.count_nonzero(tf.cast(tf.greater(pred, 0.5), tf.float32) * (tf.cast(tf.greater(y, 0.5), tf.float32) - 1)) # tf.count_nonzero(tf.argmax(pred, 1) * (tf.argmax(y, 1) - 1))
fn = tf.count_nonzero((tf.cast(tf.greater(pred, 0.5), tf.float32) - 1) * tf.cast(tf.greater(y, 0.5), tf.float32)) # tf.count_nonzero((tf.argmax(pred, 1) - 1) * tf.argmax(y, 1))

tpr = tp / tf.count_nonzero( tf.cast(tf.greater(y, 0.5), tf.float32))
fpr = fp / tf.count_nonzero(1 - tf.cast(tf.greater(y, 0.5), tf.float32)) # fp / tf.count_nonzero(1 - tf.argmax(y, 1))
fnr = fn / tf.count_nonzero(1 - tf.cast(tf.greater(y, 0.5), tf.float32)) # fn / tf.count_nonzero(1 - tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), tf.cast(tf.greater(y, 0.5), tf.float32)), tf.float32))
# Performance parameters for test phase
prob_out = pred # tf.multiply(1 - pred[:, 0], pred[:, 1])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.reduce_mean(tf.abs(tf.add(y, tf.negative(pred))) + FLAGS['Beta'] * reg) # L1

'''
Positives = tf.cast(tf.greater(y, 0.5), tf.float32) # tf.argmax(y, 1)

prob_out_positives = tf.multiply(tf.cast(prob_out, tf.float32), tf.cast(Positives, tf.float32))
number_of_high_probs = tf.cast(tf.count_nonzero(prob_out_positives), tf.float32)
values_01, indices_01 = tf.nn.top_k(prob_out_positives, k=tf.cast(tf.ceil(tf.cast(tf.count_nonzero(prob_out_positives), tf.float32) * 0.001), tf.int32))
threshold_99_9 = tf.reduce_min(values_01)

values_0, indices_0 = tf.nn.top_k(prob_out_positives, k=tf.cast(tf.ceil(tf.cast(tf.count_nonzero(prob_out_positives), tf.float32) * 0.01), tf.int32))
threshold_99_0 = tf.reduce_min(values_0)

Negatives = 1 - tf.cast(tf.greater(y, 0.5), tf.float32) # 1 - tf.argmax(y, 1)

False_detection_99_9 = tf.count_nonzero(tf.cast(tf.logical_and(tf.greater(Negatives, 0), tf.greater(prob_out, threshold_99_9)), tf.float32)) / \
                       tf.count_nonzero(Negatives)
False_detection_99_0 = tf.count_nonzero(tf.cast(tf.logical_and(tf.greater(Negatives, 0), tf.greater(prob_out, threshold_99_0)), tf.float32)) / \
                       tf.count_nonzero(Negatives)
'''

losses = []
full_data.set_batches(FLAGS['batch_size'], FLAGS['batch_size'], FLAGS['batch_size'])
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

# start creating the summary
writer = tf.summary.FileWriter('logs', sess.graph)
# Create a summary to monitor cost tensor
fake_acc_scalar = tf.placeholder(tf.float32)
fake_loss_scalar = tf.placeholder(tf.float32)
train_loss_summary_scalar = tf.summary.scalar("train loss", fake_loss_scalar)
val_loss_summary_scalar = tf.summary.scalar("val loss", fake_loss_scalar)
# Create a summary to monitor accuracy tensor
train_acc_summary_scalar = tf.summary.scalar("train_accuracy", fake_acc_scalar)
val_acc_summary_scalar = tf.summary.scalar("val_accuracy", fake_acc_scalar)
layers_histogram = [tf.summary.histogram("WC1", weights['wc1']), tf.summary.histogram("WC2", weights['wc2']),
                    tf.summary.histogram("WC3", weights['wc3']), tf.summary.histogram("WC4", weights['wc4']),
                    tf.summary.histogram("WC5", weights['wc5']), tf.summary.histogram("OUT", weights['out'])]

layers_images = [tf.summary.image("WC1", model.conv_layer_weights_transpose(weights['wc1']), max_outputs=3),
                 tf.summary.image("WC2", model.conv_layer_weights_transpose(weights['wc2']), max_outputs=3),
                 tf.summary.image("WC3", model.conv_layer_weights_transpose(weights['wc3']), max_outputs=3),
                 tf.summary.image("WC4", model.conv_layer_weights_transpose(weights['wc4']), max_outputs=3),
                 tf.summary.image("WC5", model.conv_layer_weights_transpose(weights['wc5']), max_outputs=3)
                 ]

# Merge all summaries into a single op
merged_train_summary = tf.summary.merge([train_acc_summary_scalar, train_loss_summary_scalar,
                                         layers_histogram, layers_images, feature_maps_summary])
merged_val_summary = tf.summary.merge([val_acc_summary_scalar, val_loss_summary_scalar])

# initialize the queue threads to start to shovel data
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

losses_training = []
losses_validation = []
prediction = []

epoch = 0
glob_step = 0
step = 0


if FLAGS['training']:
    while epoch < FLAGS['epochs'] and glob_step * (FLAGS['batch_size'] + 1) < FLAGS['max_iters']:
        full_data.shuffle_train()
        step = 0
        while step * (FLAGS['batch_size'] + 1) < full_data.num_of_train_samples:
            batch_x, batch_y = \
                sess.run([full_data.train_image_batch, full_data.train_label_batch])
            batch_y = batch_y.reshape((batch_y.shape[0], 1))
            # batch_y = numpy.array([1 - batch_y_temp, batch_y_temp]).transpose()
            #full_data.get_train_batch(range(step * FLAGS['batch_size'], (step + 1) * FLAGS['batch_size']))
            # Run optimization op (backprop)

            _, temp_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: FLAGS['dropout'], training: True})
            step += 1
            glob_step += 1

            if glob_step > 0 and (glob_step * (FLAGS['batch_size'] + 1)) % FLAGS['step_size'] == 0:
                FLAGS['learning_rate'] *= FLAGS['etha']

            if glob_step > 0 and step % FLAGS['display_step'] == 0:
                num_of_batches = 0
                val_loss = 0
                acc_val = 0

                train_loss = 0
                acc_train = 0
                TN = 0
                TP = 0
                FP = 0
                FN = 0
                for bat in range(int(500 / FLAGS['batch_size'])):
                    batch_train_x, batch_train_y = sess.run([full_data.train_image_batch, full_data.train_label_batch])
                    batch_train_y = batch_train_y.reshape((batch_train_y.shape[0], 1))
                    # batch_train_y = numpy.array([1 - batch_train_y_temp, batch_train_y_temp]).transpose()

                    train_loss_, tn_, tp_, fp_, fn_ = sess.run([cost, tn, tp, fp, fn],
                                        feed_dict={
                                        x: batch_train_x,
                                        y:batch_train_y,
                                        keep_prob: 1.
                                        ,training: False})
                    TN += tn_
                    TP += tp_
                    FP += fp_
                    FN += fn_
                    train_loss += train_loss_
                acc_train = (TP + TN) / (TN + TP + FP + FN)
                fpr_train = FP / (FP + TN)
                tpr_train = TP / (TP + FN)
                fnr_train = FN / (FN + TP)
                train_loss /= bat

                batch_train_x, batch_train_y = sess.run([full_data.train_image_batch, full_data.train_label_batch])
                batch_train_y = batch_train_y.reshape((batch_train_y.shape[0], 1))
                train_summary, _, _ = sess.run([merged_train_summary, fake_acc_scalar, fake_loss_scalar],
                                                                                feed_dict={fake_acc_scalar: acc_train,
                                                                                fake_loss_scalar: train_loss,
                                                                                    x: batch_train_x,
                                                                                    y: batch_train_y,
                                                                                    keep_prob: 1.
                                                                                    , training: False
                                                                                           })
                writer.add_summary(train_summary, global_step=glob_step)

                for bat in range(int(500 / FLAGS['batch_size'])):
                    batch_val_x, batch_val_y = sess.run([full_data.val_image_batch, full_data.val_label_batch])
                    batch_val_y = batch_val_y.reshape((batch_val_y.shape[0], 1))
                    # batch_val_y = numpy.array([1 - batch_val_y_temp, batch_val_y_temp]).transpose()

                    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
                        cost,tn, tp, fp, fn], feed_dict={
                        x: batch_val_x,
                        y: batch_val_y,
                        keep_prob: 1., training: False
                    })
                    TN += tn_
                    TP += tp_
                    FP += fp_
                    FN += fn_
                    val_loss += val_loss_
                acc_val = (TP + TN) / (TN + TP + FP + FN)
                fpr_val = FP / (FP + TN)
                tpr_val = TP / (TP + FN)
                fnr_val = FN / (FN + TP)
                val_loss /= bat
                # train_writer.add_summary(summary, step)
                # prediction.append(prediction_)

                val_summary, _, _ = sess.run([merged_val_summary, fake_acc_scalar, fake_loss_scalar],
                                                                feed_dict={fake_acc_scalar: acc_val,
                                                                           fake_loss_scalar: val_loss})
                writer.add_summary(val_summary, global_step=glob_step)

                losses_validation.append(val_loss)
                losses_training.append(train_loss)

                print("\rEpoch " + str(epoch) + ", Iter " + str(step*FLAGS['batch_size'])
                      + "\nTraining Loss = {:.6f}".format(train_loss) +
                      " \nAccuracy = {:.6f}".format(acc_train)+
                      " \nFPR = {:.6f}".format(fpr_train) +
                      " \nTPR = {:.6f}".format(tpr_train) +
                      " \nFNR = {:.6f}".format(fnr_train), flush=True)

                print("\nValidation Loss = " + "{:.6f}".format(val_loss) +
                      " \nAccuracy = {:.6f}".format(acc_val) +
                      " \nFPR = {:.6f}".format(fpr_val) +
                      " \nTPR = {:.6f}".format(tpr_val) +
                      " \nFNR = {:.6f}".format(fnr_val))

                # save_path = saver.save(sess, r"E:\studies\NetworkSeg\checkpoints\model.ckpt")
                # print("Model saved in file: %s" % save_path)
            print('\rTOTAL PROGRESS' + ('......' if step % 2 else '..... ') + '{:3d}% done epochs {:d} / {:d} with {:3d}% progress'.format(
                int(100 * glob_step * (FLAGS['batch_size'] + 1) / (FLAGS['epochs'] * full_data.num_of_train_samples)), epoch, FLAGS['epochs'],
                int(100 * step * (FLAGS['batch_size'] + 1) / (full_data.num_of_train_samples))),
                  end='', flush=True)
        epoch += 1


if FLAGS['training']:
    save_path = saver.save(sess, FLAGS['output_model_path'])
    print("Final model saved in file: %s" % save_path)
    # sess.close()
'''
test_acc = sess.run(test_accuracy, feed_dict={x: full_data.test_x[:100], y: full_data.test_y[:100], keep_prob: 1.,
                                              training: False})
train_acc = sess.run(test_accuracy, feed_dict={x: full_data.train_x[:100], y: full_data.train_y[:100], keep_prob: 1.,
                                               training: False})
val_acc = sess.run(test_accuracy, feed_dict={x: full_data.val_x[:100], y: full_data.val_y[:100], keep_prob: 1.,
                                             training: False})

print("train accuracy = {:f} val accuracy = {:f} and test accuracy = {:f}".format(train_acc, val_acc, test_acc))
'''

TN = 0
TP = 0
FP = 0
FN = 0
for bat in range(int(full_data.num_of_train_samples / FLAGS['batch_size'])):
    batch_x, batch_y = \
        sess.run([full_data.train_image_batch, full_data.train_label_batch])
    batch_y = batch_y.reshape((batch_y.shape[0], 1))
    # batch_y = numpy.array([1 - batch_y_temp, batch_y_temp]).transpose()

    train_loss_, tn_, tp_, fp_, fn_ = sess.run([cost, tn, tp, fp, fn],
                                     feed_dict={
                        x: batch_x,
                        y:batch_y,
                        keep_prob: 1.
                        ,training: False})
    TN += tn_
    TP += tp_
    FP += fp_
    FN += fn_
acc_train = (TP + TN) / (TN + TP + FP + FN)
fpr_train = FP / (FP + TN)
tpr_train = TP / (TP + FN)
fnr_train = FN / (FN + TP)


TN = 0
TP = 0
FP = 0
FN = 0

for bat in range(int(full_data.num_of_val_samples / FLAGS['batch_size'])):

    batch_x, batch_y = \
        sess.run([full_data.val_image_batch, full_data.val_label_batch])
    batch_y = batch_y.reshape((batch_y.shape[0], 1))
    # batch_y = numpy.array([1 - batch_y_temp, batch_y_temp]).transpose()

    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
        cost, tn, tp, fp, fn], feed_dict={
        x: batch_x,
        y: batch_y,
        keep_prob: 1., training: False
    })
    TN += tn_
    TP += tp_
    FP += fp_
    FN += fn_
acc_val = (TP + TN) / (TN + TP + FP + FN)
fpr_val = FP / (FP + TN)
tpr_val = TP / (TP + FN)
fnr_val = FN / (FN + TP)

TN = 0
TP = 0
FP = 0
FN = 0

for bat in range(int(full_data.num_of_test_samples / FLAGS['batch_size'])):
    batch_x, batch_y = \
        sess.run([full_data.test_image_batch, full_data.test_label_batch])
    batch_y = batch_y.reshape((batch_y.shape[0], 1))
    # batch_y = numpy.array([1 - batch_y_temp, batch_y_temp]).transpose()

    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
        cost, tn, tp, fp, fn], feed_dict={
        x: batch_x,
        y: batch_y,
        keep_prob: 1., training: False
    })
    TN += tn_
    TP += tp_
    FP += fp_
    FN += fn_

acc_test = (TP + TN) / (TN + TP + FP + FN)
fpr_test = FP / (FP + TN)
tpr_test = TP / (TP + FN)
fnr_test = FN / (FN + TP)

print("train accuracy = {:f} val accuracy = {:f} and test accuracy = {:f}".format(acc_train, acc_val, acc_test))

# stop our queue threads and properly close the session
coord.request_stop()
coord.join(threads)