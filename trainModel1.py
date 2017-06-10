import tensorflow as tf
import model
import data
import matplotlib.pyplot as plt


FLAGS = dict()
FLAGS['patch_size'] = 100
FLAGS['learning_rate'] = 0.005
FLAGS['kernel_size'] = 5
FLAGS['num_of_filters1'] = 50
FLAGS['num_of_filters2'] = 100
FLAGS['num_of_filters3'] = 50
FLAGS['batch_size'] = 20
FLAGS['max_iters'] = 10000000
FLAGS['epochs'] = 50
FLAGS['dropout'] = 0.75
FLAGS['display_step'] = 50
FLAGS['pre_trained_model_path'] = './10_06_17__1.model.ckpt'
FLAGS['output_model_path'] = './10_06_17__1.model.ckpt'
FLAGS['train_data_path'] = './images/train/train.pickle'
FLAGS['val_data_path'] = './images/val/val.pickle'
FLAGS['test_data_path'] = './images/test/test.pickle'

# load data
full_data = data.Data('')
full_data.import_data(FLAGS['train_data_path'], FLAGS['val_data_path'], FLAGS['test_data_path'])

"""  Create all placeholders of the Net  """
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, FLAGS['patch_size'], FLAGS['patch_size'], 3])
y = tf.placeholder(tf.float32, shape=[None, 2])

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
    'wc4': tf.Variable(tf.random_normal([FLAGS['num_of_filters3'] * 2, 200], 0, 1.)),
    'wc5': tf.Variable(tf.random_normal([200, 200], 0, 1.)),
    'out': tf.Variable(tf.random_normal([200, 2], 0, 1.))
}
#variable_summaries(weights)
biases = {
    'bc4': tf.Variable(tf.random_normal([200])),
    'bc5': tf.Variable(tf.random_normal([200])),
    'out': tf.Variable(tf.random_normal([2]))
}

# initialize the CNN
pred = model.net(x, weights, biases, FLAGS, True, keep_prob)
test_pred = model.net(x, weights, biases, FLAGS, False, keep_prob)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.reduce_mean(tf.abs(tf.add(y, tf.negative(pred)))) # L1
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS['learning_rate']).minimize(cost)
losses = []

# Strat session for training

print("Start a session")
sess = tf.Session()
print("Open a saver")
saver = tf.train.Saver()

print("Try to load pre trained model")
try:
    saver.restore(sess, FLAGS['pre_trained_model_path'])
    print("Load previous model")
except:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("Could not load model, use model initialization")

losses_training = []
losses_validation = []
prediction = []

epoch = 0
global_step = 0
step = 0
while epoch < FLAGS['epochs'] and step < FLAGS['max_iters']:
    full_data.shuffle_train()
    step = 0
    while step * (FLAGS['batch_size'] + 1) < full_data.train_x.shape[0]:
        batch_x, batch_y = full_data.get_train_batch(range(step * FLAGS['batch_size'], (step + 1) * FLAGS['batch_size']))
        # Run optimization op (backprop)
        sess.run([optimizer], feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: FLAGS['dropout']})

        step += 1
        global_step += 1

        if step % FLAGS['display_step'] == 0:
            #summary, loss = sess.run([merged, cost], feed_dict={x: batch_x,
            #                                                  y: batch_y,
            #                                                  keep_prob: 1.})

            train_loss, acc_train = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            val_loss, acc_val = sess.run([cost, accuracy], feed_dict={
                x: full_data.val_x[:100], y: full_data.val_y[:100], keep_prob: 1.
            })
            # train_writer.add_summary(summary, step)
            # prediction.append(prediction_)
            losses_validation.append(val_loss)
            losses_training.append(train_loss)

            print("Epoch " + str(epoch) + ", Iter " + str(step*FLAGS['batch_size']) + ", Validation Loss = " + "{:.6f}".format(val_loss)
                  + ", Validation Accuracy = " + "{:.6f}".format(acc_val) +
                  " Training Loss = {:.6f}".format(train_loss) +
                  " Training Accuracy = {:.6f}".format(acc_train))
            # save_path = saver.save(sess, r"E:\studies\NetworkSeg\checkpoints\model.ckpt")
            # print("Model saved in file: %s" % save_path)
    epoch += 1


save_path = saver.save(sess, FLAGS['output_model_path'])
print("Final model saved in file: %s" % save_path)

test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_pred, 1), tf.argmax(y, 1)), tf.float32))

test_acc = sess.run(test_accuracy, feed_dict={x: full_data.test_x, y: full_data.test_y, keep_prob: 1.})
train_acc = sess.run(test_accuracy, feed_dict={x: full_data.train_x, y: full_data.train_y, keep_prob: 1.})
val_acc = sess.run(test_accuracy, feed_dict={x: full_data.val_x, y: full_data.val_y, keep_prob: 1.})

print("train accuracy = {:f} val accuracy = {:f} and test accuracy = {:f}".format(train_acc, val_acc, test_acc))
plt.plot(losses_validation)
plt.plot(losses_training)
plt.show()
