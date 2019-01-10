import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt


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


# Build network
n_inputs = 32 * 32 * 3
n_outputs = 10

#X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#y = tf.placeholder(tf.int64, shape=(None), name="y")


''' This was adapted from TensorFlow's website:

        https://www.tensorflow.org/tutorials/layers
'''
def my_first_cnn(features, labels, mode):
    # shaped = tf.cast(tf.transpose(tf.reshape(features['X'], [-1, 3, 32, 32]), (0, 2, 3, 1)),tf.float32)
    # n_filters1 = 32
    # conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
    # pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
    # flat = tf.reshape(pool1, [-1, 16 * 16 * n_filters1])
    # n_hidden1 = 1024
    # hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.elu)
    # logits = tf.layers.dense(hidden1, 10, name="outputs")

    shaped = tf.cast(tf.transpose(tf.reshape(features['x'], [-1, 3, 32, 32]), (0, 2, 3, 1)),tf.float32)
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
    logits = tf.layers.dense(hidden1, units=10)
    predictions = {
        "classes": tf.argmax(input=logits,axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)


    #Loss Calculations for training and eval
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions, name="accuracy")
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    print("hi there");
    '''this code was taken from Peter Drake's lecture'''
    print("starting")
    data_dir = "C:/Users/Test/code/AI/projects/deeplearning/cifar-10-batches-py/"
    train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in range(1,5)]
    X_train = np.concatenate([t[b'data'] for t in train],axis=0)
    y_train = np.array(list(itertools.chain(*[t[b'labels']for t in train])))
    valid = unpickle(data_dir + 'data_batch_5')
    X_valid = valid[b'data']
    y_valid = np.array(valid[b'labels'])
    labels = unpickle(data_dir + 'batches.meta')[b'label_names']
 
    classifier = tf.estimator.Estimator(model_fn=my_first_cnn,
        model_dir="C:/Users/Test/code/AI/projects/deeplearning/modeldir/")


    # Set up logging for predictions
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_valid},
    y=y_valid,
    num_epochs=1,
    shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()