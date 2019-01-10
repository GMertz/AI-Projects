import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt

data_dir = "/home/users/gmertz/cifar-10-batches-py/"
save_dir = "/home/users/gmertz/modeldir/"
#data_dir = "C:/Users/Test/code/AI/projects/deeplearning/cifar-10-batches-py/"
#save_dir = "C:/Users/Test/code/AI/projects/deeplearning/modeldir/"

def unpickle(file):
    '''adapted from the CIFAR page: https://www.cs.toronto.edu/~kriz/cifar.html '''
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def gimmeData():
    train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in range(1,5)]
    valid = unpickle(data_dir + 'data_batch_5')
    return {
     "X_train": np.concatenate([t[b'data'] for t in train],axis=0),
     "y_train": np.array(list(itertools.chain(*[t[b'labels']for t in train]))),
     "X_valid": valid[b'data'],
     "y_valid": np.array(valid[b'labels']),
     "labels": unpickle(data_dir + 'batches.meta')[b'label_names']
    }

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(total_images, size=batch_size)  # not shown
    X_image = X_train[indices] # not shown
    y_label = y_train[indices] # not shown
    return X_image, y_label


# Build network
n_inputs = 32 * 32 * 3
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))
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
    logits = tf.layers.dense(hidden1, units=10,name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    right = tf.nn.in_top_k(logits,y,1)
    #accuracy = tf.metrics.accuracy(labels=y, predictions=predictions name="accuracy")
    accuracy = tf.reduce_mean(tf.cast(right, tf.float32))

    saver = tf.train.Saver()
    batch_size = 100
    batch_nums = 40000 // batch_size
    n_epochs = 50
    data = gimmeData()
    X_train = np.array([data["X_train"][(i*batch_size):(i+1 * batch_size)] for i in range(batch_nums) ])
    y_train = np.array([data["y_train"][(i*batch_size):(i+1 * batch_size)] for i in range(batch_nums) ])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for n in range(n_epochs):
        X_bat = None
        y_bat = None
        for i in range(batch_nums-1):
            X_bat = X_train[i]
            y_bat = y_train[i]
            sess.run(train_op, feed_dict={X:X_bat,y:y_bat})
        train_acc = accuracy.eval(feed_dict={X:X_bat,y:y_bat})
        val_acc = accuracy.eval(feed_dict={X:data["X_valid"],y:data["y_valid"]})
        print("Epoch: " + str(n) + "\n \t train: " + str(train_acc) + "\n \t val: " + str(val_acc))


    saver.save(sess, save_dir + "model.cpkt")


