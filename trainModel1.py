import tensorflow as tf
import model
import data
import numpy
import matplotlib.pyplot as plt


FLAGS = dict()
FLAGS['patch_size'] = 224
FLAGS['learning_rate'] = 0.0001
FLAGS['kernel_size'] = 3
FLAGS['num_of_filters1'] = 32
FLAGS['num_of_filters2'] = 64
FLAGS['num_of_filters3'] = 128
FLAGS['num_of_filters4'] = 256
FLAGS['num_of_filters5'] = 512
FLAGS['batch_size'] = 20
FLAGS['max_iters'] = 10000000
FLAGS['epochs'] = 50
FLAGS['training'] = True
FLAGS['dropout'] = 0.75
FLAGS['display_step'] = 100
FLAGS['pre_trained_model_path'] = './models/07_07_17__8.model.ckpt' #'./models/07_07_17__6.model.ckpt'
FLAGS['output_model_path'] = './models/07_07_17__9.model.ckpt'
FLAGS['train_data_path'] = './images/train/train.pickle'
FLAGS['val_data_path'] = './images/val/val.pickle'
FLAGS['test_data_path'] = './images/test/test.pickle'
FLAGS['Beta'] = 0.001
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
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters2'], FLAGS['num_of_filters3']]
        , 0, 1.)),

    'wc4': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters3'], FLAGS['num_of_filters4']]
        , 0, 1.)),

    'wc5': tf.Variable(tf.random_normal(
        [FLAGS['kernel_size'], FLAGS['kernel_size'], FLAGS['num_of_filters4'], FLAGS['num_of_filters5']]
        , 0, 1.)),

    'wc6': tf.Variable(tf.random_normal(
        [int(FLAGS['num_of_filters5'] * numpy.round(float(FLAGS['patch_size'])/(2**6))**2), 512]
        , 0, 1.)),
    # 'wc4': tf.Variable(tf.random_normal([FLAGS['num_of_filters3'] * 2, 200], 0, 1.)),
    # 'wc5': tf.Variable(tf.random_normal([200, 200], 0, 1.)),

    'out': tf.Variable(tf.random_normal([18432, 2], 0, 1.))
}
#variable_summaries(weights)
biases = {
    'bc6': tf.Variable(tf.zeros((18432))),
    'out': tf.Variable(tf.zeros((2)))
}

# initialize the CNN
pred, reg = model.net(x, weights, biases, FLAGS, training, keep_prob)
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
cost = tf.reduce_mean(tf.abs(tf.add(y, tf.negative(pred))) + FLAGS['Beta'] * reg) # L1

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

                    train_loss_, tn_, tp_, fp_, fn_ = sess.run([cost, tn, tp, fp, fn],
                                                     feed_dict={
                                        x: full_data.train_x[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
                                        y:full_data.train_y[bat * FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
                                        keep_prob: 1.
                                        ,training: True})
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

                for bat in range(int(500 / FLAGS['batch_size'])):

                    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
                        cost,tn, tp, fp, fn], feed_dict={
                        x: full_data.val_x[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
                        y: full_data.val_y[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
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

                losses_validation.append(val_loss)
                losses_training.append(train_loss)

                print("Epoch " + str(epoch) + ", Iter " + str(step*FLAGS['batch_size'])
                      + "\nTraining Loss = {:.6f}".format(train_loss) +
                      " \nAccuracy = {:.6f}".format(acc_train)+
                      " \nFPR = {:.6f}".format(fpr_train) +
                      " \nTPR = {:.6f}".format(tpr_train) +
                      " \nFNR = {:.6f}".format(fnr_train))

                print("\nValidation Loss = " + "{:.6f}".format(val_loss) +
                      " \nAccuracy = {:.6f}".format(acc_val) +
                      " \nFPR = {:.6f}".format(fpr_val) +
                      " \nTPR = {:.6f}".format(tpr_val) +
                      " \nFNR = {:.6f}".format(fnr_val))

                # save_path = saver.save(sess, r"E:\studies\NetworkSeg\checkpoints\model.ckpt")
                # print("Model saved in file: %s" % save_path)
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
for bat in range(int(full_data.train_y.shape[0] / FLAGS['batch_size'])):

    train_loss_, tn_, tp_, fp_, fn_ = sess.run([cost, tn, tp, fp, fn],
                                     feed_dict={
                        x: full_data.train_x[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
                        y:full_data.train_y[bat * FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
                        keep_prob: 1.
                        ,training: True})
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

for bat in range(int(full_data.train_y.shape[0] / FLAGS['batch_size'])):
    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
        cost, tn, tp, fp, fn], feed_dict={
        x: full_data.val_x[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
        y: full_data.val_y[bat*FLAGS['batch_size']:(bat + 1)*FLAGS['batch_size']],
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

for bat in range(int(full_data.test_y.shape[0] / FLAGS['batch_size'])):
    val_loss_, tn_, tp_, fp_, fn_ = sess.run([
        cost, tn, tp, fp, fn], feed_dict={
        x: full_data.test_x[bat * FLAGS['batch_size']:(bat + 1) * FLAGS['batch_size']],
        y: full_data.test_y[bat * FLAGS['batch_size']:(bat + 1) * FLAGS['batch_size']],
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