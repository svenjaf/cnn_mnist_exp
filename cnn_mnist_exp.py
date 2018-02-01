"""cnn_mnist_exp.py

Svenja Fleischer, 2018

Experimenting with TensorFlow - implement a simple convolutional
network for MNIST classification.

Includes two convolution / max-pooling layers, a fully connected
hidden layer and a softmax output layer. Uses L2 regularization and
a log-likelihood cost function. Parameters taken from Michael
Nielsen, "Neural Networks and Deep Learning", Chapter 6.
"""

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # some parameters:
    LEARNING_RATE = 0.03
    REG_PARAM = 0.1/50000

    # define the convolutional network:
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=REG_PARAM)
    conv_layer1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=20,
            kernel_size=[5, 5],
            kernel_regularizer=l2_regularizer,
            padding="valid",
            activation=tf.nn.relu)
    pool_layer1 = tf.layers.max_pooling2d(
            inputs=conv_layer1,
            pool_size=[2, 2],
            strides=2)
    conv_layer2 = tf.layers.conv2d(
            inputs=pool_layer1,
            filters=40,
            kernel_size=[5, 5],
            kernel_regularizer=l2_regularizer,
            padding="valid",
            activation=tf.nn.relu)
    pool_layer2 = tf.layers.max_pooling2d(
            inputs=conv_layer2,
            pool_size=[2, 2],
            strides=2)
    flattening_layer = tf.reshape(pool_layer2, [-1, 40 * 4 * 4])
    fc_layer = tf.layers.dense(
            inputs=flattening_layer,
            units=100,
            kernel_regularizer=l2_regularizer,
            activation=tf.nn.relu)
    #dropout = tf.layers.dropout(
    #        inputs=fc_layer,
    #        rate=0.5,
    #        training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(
            inputs=fc_layer, # dropout,
            units=10,
            kernel_regularizer=l2_regularizer)

    predictions = {
            "class": tf.argmax(logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax")
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #loss = (tf.losses.sparse_softmax_cross_entropy(
        #labels=labels, logits=logits)
        #+ tf.losses.get_regularization_loss())

    # convert labels to one-hot encoding:
    labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), 10)
    # log likelihood cost function + L2 regularization:
    loss = -tf.reduce_mean(
            input_tensor=tf.log(tf.reduce_sum(
                input_tensor=tf.multiply(
                    labels_one_hot,
                    predictions["probabilities"]),
                axis=1))) + tf.losses.get_regularization_loss()

    accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predictions["class"])
    tf.summary.scalar("model_accuracy", accuracy[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)

    evaluation_op = { "accuracy" : accuracy }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
            eval_metric_ops=evaluation_op)


def main():
    # parameters:
    BATCH_SIZE = 100
    NUM_TRAINING_EPOCHS = 60
    MODEL_DIR = "/home/svenja/src/practice/ml/tensorflow/cnn_mnist_exp/log"

    cnn_mnist = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=MODEL_DIR)

    # get MNIST data:
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    training_data = mnist.train.images
    training_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # logging:
    what_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=what_to_log,
            every_n_iter=1000)
    
    training_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": training_data},
            y=training_labels,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_TRAINING_EPOCHS,
            shuffle=True)

    cnn_mnist.train(
            input_fn=training_input_fn,
            hooks=[logging_hook])

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_labels,
            batch_size=BATCH_SIZE,
            num_epochs=1,
            shuffle=False)

    test_result = cnn_mnist.evaluate(input_fn=test_input_fn)
    print(test_result)

if __name__ == "__main__":
    main()

