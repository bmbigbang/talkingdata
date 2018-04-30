from datetime import datetime
import numpy as np
import tensorflow as tf
import time
from matplotlib import pyplot as plt

# read train file line by line
train = []
labels = []
labels_1 = []
norm_vals = [0, 0, 0, 0, 0, 0]
with open('train.csv', 'r') as file_object:
    line = file_object.readline()
    count = 0
    # skip header
    if line.split(",")[0][0].isalpha():
        line = file_object.readline()
    while line:
        row = line.split(",")
        time_delta = 0
        if row[6]:
            time_start = datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
            time_end = datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S')
            time_delta = time_end - time_start
            time_delta = int(time_delta.total_seconds())

        train.append(list(map(int, row[:5])) + [time_delta])
        for i, j in enumerate(train[-1]):
            if norm_vals[i] < j:
                norm_vals[i] = j
        label = int(row[7])
        labels.append(label)
        if label:
            labels_1.append(count)

        line = file_object.readline()
        count += 1


norm_vals = np.array(norm_vals)
train = np.array(train)
labels = np.array(labels)
labels_1 = np.array(labels_1)

# since the fraudulent labels are a very small proportion of the data, we will resample with more
# positive labels to learn to distinguish these features better
print(len(labels_1) / count)


# construct batch generator which prefers to include fraudulent labels
def get_batches(train_list, batch_size):
    """ Create a generator of row batches as a tuple (inputs, targets) """

    n_batches = count // batch_size

    # only full batches
    X = train_list[:n_batches * batch_size]

    for idx in range(0, len(X), batch_size):
        batch = X[idx:idx + batch_size]
        x = []
        for ii in range(len(batch)):
            if np.random.rand() >= 0.8:
                batch_x = train_list[labels_1[np.random.randint(len(labels_1))]]
                x.append((batch_x / norm_vals) + 0.0001)
            else:
                batch_x = batch[ii]
                x.append((batch_x / norm_vals) + 0.0001)

        yield x, x


train_graph = tf.Graph()
# Size of the encoding layer (the hidden layer)
encoding_dim = 100
learning_rate = 0.00001
with train_graph.as_default():
    # Input and target placeholders
    inp_shape = train[0].shape[0]

    is_train = tf.placeholder(tf.bool, name='train_cond')
    inputs_ = tf.placeholder(tf.float32, (None, inp_shape), name='inputs')
    targets_ = tf.placeholder(tf.float32, (None, inp_shape), name='targets')

    # Output of hidden layer, single fully connected layer here with ReLU activation
    encoded = tf.layers.dense(inputs_, encoding_dim)
    norm = tf.layers.batch_normalization(encoded, training=is_train)
    norm = tf.maximum(0.1 * norm, norm)

    # Output layer logits, fully connected layer with no activation
    logits = tf.layers.dense(norm, inp_shape, activation=None)
    # Sigmoid output from logits
    decoded = tf.nn.sigmoid(logits, name='outputs')

    # Sigmoid cross-entropy loss
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
    # Mean of the loss
    cost = tf.reduce_mean(loss)

    # Adam optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# !mkdir checkpoints

epochs = 10
batch_size = 1000

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train, batch_size)
        start = time.time()
        for x, y in batches:

            feed = {inputs_: x,
                    targets_: y,
                    is_train: True}
            train_loss, _ = sess.run([cost, opt], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")


with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    samples_1 = np.take(train, labels_1, axis=0)
    samples_1 = np.array([(i / norm_vals) + 0.0001 for i in samples_1])
    embed_mat_1 = sess.run(encoded, feed_dict={inputs_: samples_1, is_train: False})

    generalization = np.sum(embed_mat_1, axis=0)

    generalization = generalization / np.sum(generalization)

    train_0 = []
    for i in range(200000):
        idx = np.random.randint(0, count)
        if labels[idx] != 0:
            train_0.append(idx)

    samples_0 = np.take(train, train_0, axis=0)
    samples_0 = np.array([(i / norm_vals) + 0.0001 for i in samples_0])
    embed_mat_0 = sess.run(encoded, feed_dict={inputs_: samples_0, is_train: False})
    generalization_0 = np.sum(embed_mat_0, axis=0)

    generalization_0 = generalization_0 / np.sum(generalization_0)


#plt.plot(generalization)
#plt.plot(generalization_0)
#plt.show()

# find max difference dimensions
