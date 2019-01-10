import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt

data_dir = "C:/Users/Test/code/AI/projects/deeplearning/cifar-10-batches-py/"
save_dir = "C:/Users/Test/code/AI/projects/deeplearning/modeldir/"


tf.logging.set_verbosity(tf.logging.INFO)

def unpickle(file):
    '''adapted from the CIFAR page: https://www.cs.toronto.edu/~kriz/cifar.html '''
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def viewImage(i):
    test = X_valid.reshape(10000,3,32,32)
    img = test[i].T
    #img[:,[0,1]] = img[:,[1,0]]
    plt.imshow(img)
    plt.show()
    print(labels[y_valid[i]])

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


# Build network
n_inputs = 32 * 32 * 3

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("cnn"):
    shaped = tf.cast(tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1)),tf.float32)
    n_filters1 = 32
    conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
    n_filters2 = 64
    conv2 = tf.layers.conv2d(pool1, n_filters2, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='valid')
    n_filters3 = 128
    conv3 = tf.layers.conv2d(pool2, n_filters3, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2, padding='valid')
    flat = tf.reshape(pool3, [-1, 4 * 4 * 128])
    n_hidden1 = 1024
    hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.elu)
    logits = tf.layers.dense(hidden1, 10)

with tf.name_scope("loss"):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, reduction=tf.losses.Reduction.MEAN)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    predictions = tf.argmax(input=logits,axis=1)
    labels = y
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
    #accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))


    saver = tf.train.Saver()
    batch_size = 10
    batch_nums = 100 // batch_size
    n_epochs = 1
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
        train_acc = accuracy[0].eval(feed_dict={X:X_train[0],y:y_train[0]})
        val_acc = accuracy[0].eval(feed_dict={X:data["X_valid"],y:data["y_valid"]})
        print("Epoch: " + str(n) + "\n \t train: " + str(train_acc) + "\n \t val: " + str(val_acc))


    saver.save(sess, save_dir + "model.cpkt")


