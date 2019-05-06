"""
Accuracy:0.7448
"""
import numpy as np
import tensorflow as tf
from dataset import load_data
from dataset import load_text_dataset
from util import plot_confusion_matrix
import matplotlib.pyplot as plt

NUM_EPOCHS = 20
BATCH_SIZE = 64
LR = 0.01
N_CLASS = 50
N_WORD = 4464


def rnn_model(features):
    embedding = tf.Variable(tf.random_uniform([4464, 256], -1.0, 1.0), trainable=True)
    embed = tf.nn.embedding_lookup(embedding, features)
    cell = tf.nn.rnn_cell.GRUCell(50)
    rnn_out, state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    last_output = rnn_out[:, -1]
    dense = tf.layers.dense(inputs=last_output, units=50, activation=tf.nn.relu)

    return dense

def main():
    # Load dataset
    train_data, test_data = load_text_dataset(path='./question/', seq_len=5)
    train_x, train_y = train_data
    test_x, test_y = test_data

    # placeholder for input variables
    x_placeholder = tf.placeholder(tf.int32,
                                   shape=(BATCH_SIZE, None))
    y_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))

    # get the loss function and the prediction function for the network
    pred_op = rnn_model(x_placeholder)
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder,
                                                     logits=pred_op)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(LR)
    train_op = optimizer.minimize(loss_op)

    # start tensorflow session
    sess = tf.Session()

    # initialization
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess.run(init)
    save_path = saver.save(sess, "question_gru_model/model.ckpt")


    # train loop -----------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        n_batch = 0
        for i in range(0, train_x.shape[0]-BATCH_SIZE, BATCH_SIZE):
            # get batch data
            x_batch = train_x[i:i+BATCH_SIZE]
            y_batch = train_y[i:i+BATCH_SIZE]

            # run step of gradient descent
            feed_dict = {
                x_placeholder: x_batch,
                y_placeholder: y_batch,
            }
            _, loss_value = sess.run([train_op, loss_op],
                                     feed_dict=feed_dict)

            running_loss += loss_value
            n_batch += 1

        print('[Epoch: %d] loss: %.3f' %
              (epoch + 1, running_loss / (n_batch)))

    # test loop -----------------------------------------------------
    all_predictions = np.zeros((0, 1))
    actual_size = 1472
    for i in range(0, 192, BATCH_SIZE):
        x_batch = test_x[i:i+BATCH_SIZE]

        # pad small batch
        # padded = BATCH_SIZE - x_batch.shape[0]
        # if padded > 0:
        #     x_batch = np.pad(x_batch,
        #                      ((0, padded), (0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),
        #                       (0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),
        #                       (0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),
        #                       (0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),
        #                       (0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)),
        #                      'constant')

        # run step
        feed_dict = {x_placeholder: x_batch}
        batch_pred = sess.run(pred_op,
                              feed_dict=feed_dict)

        # recover if padding
        # if padded > 0:
        #     batch_pred = batch_pred[0:-padded]

        # get argmax to get class prediction
        batch_pred = np.argmax(batch_pred, axis=1)

        all_predictions = np.append(all_predictions, batch_pred)
    cnt = 0.0
    for i in xrange(len(all_predictions)):
        if all_predictions[i] == test_y[i]:
            cnt += 1

    print cnt/len(all_predictions)


if __name__ == "__main__":
    main()