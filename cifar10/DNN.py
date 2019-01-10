import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt

#data_dir = "/home/users/gmertz/cifar-10-batches-py/"
#path to cifar 10 data


def unpickle(file):
    """Adapted from the CIFAR page: http://www.cs.utoronto.ca/~kriz/cifar.html"""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def fetch_batch(epoch, batch_index, batch_size):
    '''   adapted from the book's jupyter notebooks
    https://github.com/ageron/handson-ml/blob/master/09_up_and_running_with_tensorflow.ipynb'''
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(total, size=batch_size)
    X_data = X_train[indices]
    y_data = y_train[indices]
    return X_data, y_data


# Prepare data
train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
X_train = np.concatenate([t[b'data'] for t in train], axis=0)
y_train = np.array(list(itertools.chain(*[t[b'labels'] for t in train])))
valid = unpickle(data_dir + 'data_batch_5')
X_valid = valid[b'data']
y_valid = np.array(valid[b'labels'])

# Build network
''' this section was adapted from Peter Drake's code  '''
n_inputs = 32 * 32 * 3
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

'''
Here we create our models structure:
Input layer,
convolutional layer 1 - 32 filters, kernal_size 3, "same" padding and stride = 1, so image stays the same size
pool layer 1 - reduces the size of the (stride 2, no padding)
conv. layer 2 - similar to the first, but with 64 filters now
pool layer 2 - same purpose of the previous pooling layer
conv. layer 3 - 128 filters this time
pool layer 3 - reduces the size once more, now the image is 4x4, with 128 filters
flat - reshapes the previous layer to be put into the dense layer
dense layer - 1024 neuron fully connected layer
logits - output layer with models guess'
'''
with tf.name_scope("dnn"):
    shaped = tf.cast(tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1)), tf.float32)
    n_filters1 = 32
    conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
    n_filters2 = 64
    conv2 = tf.layers.conv2d(pool1, n_filters2, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='valid')
    n_filters3 = 128
    conv3 = tf.layers.conv2d(pool2, n_filters3, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='valid')
    flat = tf.reshape(pool3, [-1, 4 * 4 * n_filters3])
    n_hidden1 = 1024
    hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.elu)
    logits = tf.layers.dense(hidden1, units=10)

# here we calculate the loss, to be used in training the model (used by training_op)
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# here we define the training_op. We use Adam Optimization and learning rate of .001
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=.001)
    training_op = optimizer.minimize(loss)

# here we define our accuracy metric which we use to measure our model's performance
with tf.name_scope("eval"):
    correct = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

batch_size = 1000
total = 40000
epochs = 50
n_batches = total // batch_size

with tf.Session() as sess:  # start tensorflow session
    sess.run(init)  # initialize tensorflow variables
    for epoch in range(epochs):
        for batch_index in range(n_batches):
            X_image, y_label = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_image, y: y_label})

    acc_train = sess.run(accuracy, feed_dict={X: X_train, y: y_train})
    acc_val = sess.run(accuracy, feed_dict={X: X_valid, y: y_valid})
    print("Epoch: " + str(epoch) + "\n \t val: " + str(acc_val) + "\n \t train: " + str(acc_train))

