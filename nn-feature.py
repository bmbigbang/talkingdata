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
def get_batches(train_list, lab, batch_size, random_bias=True):
    """ Create a generator of row batches as a tuple (inputs, targets) """

    n_batches = count // batch_size

    # only full batches
    X = train_list[:n_batches * batch_size]

    for idx in range(0, len(X), batch_size):
        batch = X[idx:idx + batch_size]
        x, y = [], []
        if random_bias:
            for ii in range(len(batch)):
                if np.random.rand() >= 0.8:
                    batch_x = train_list[labels_1[np.random.randint(len(labels_1))]]
                    x.append((batch_x / norm_vals) + 0.0001)
                else:
                    batch_x = batch[ii]
                    x.append((batch_x / norm_vals) + 0.0001)
            y = x
        else:
            for ii in range(len(batch)):
                batch_x = batch[ii]
                x.append((batch_x / norm_vals) + 0.0001)
                y.append(lab[idx + ii])

        yield x, y


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
    encoded = tf.layers.dense(inputs_, encoding_dim, name='encoded')
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
        batches = get_batches(train, labels, batch_size)
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
    save_path = saver.save(sess, "checkpoints/encoder.ckpt")


with train_graph.as_default():
    saver = tf.train.Saver()


with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    samples_1 = np.take(train, labels_1, axis=0)
    samples_1 = np.array([(i / norm_vals) + 0.0001 for i in samples_1])
    embed_mat_1 = sess.run(encoded, feed_dict={inputs_: samples_1, is_train: False})

    generalization = np.sum(embed_mat_1, axis=0)

    generalization = generalization / np.sum(generalization)
    for i in range(0, len(train), batch_size):
        samples_0 = np.array([(j / norm_vals) + 0.0001 for j in train[i:i + batch_size]])
        embed_mat_0 = sess.run(encoded, feed_dict={inputs_: samples_0, is_train: False})
        np.save('/home/ubuntu/talkingdata/data/embed-{}.npy'.format(i / batch_size), embed_mat_0)

    generalization_0 = np.sum(embed_mat_0, axis=0)

    generalization_0 = generalization_0 / np.sum(generalization_0)


with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    ## tf.all_variables()
    embed_mat_w1 = train_graph.get_tensor_by_name('encoded/kernel:0').eval()
    embed_mat_b1 = train_graph.get_tensor_by_name('encoded/bias:0').eval()


#plt.plot(generalization)
#plt.plot(generalization_0)
#plt.show()

# find max difference dimensions
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(min_samples_split=40)
# labels_0 = np.take(labels, train_0, axis=0)
# clf.fit(embed_mat_0 - generalization, labels_0)


# define the decision making model
learning_rate = 0.0001
with train_graph.as_default():
    # Input and target placeholders

    is_train_2 = tf.placeholder(tf.bool, name='train_cond_2')
    inputs_2 = tf.placeholder(tf.float32, (None, encoding_dim), name='inputs_2')
    targets_2 = tf.placeholder(tf.float32, (None, 1), name='targets_2')

    # Output of hidden layer, single fully connected layer here with ReLU activation
    x1 = tf.layers.dense(inputs_2, 60)
    norm1 = tf.layers.batch_normalization(x1, training=is_train_2)
    norm1 = tf.layers.dropout(norm1, 0.2)
    norm1 = tf.maximum(0.1 * norm1, norm1)

    x2 = tf.layers.dense(norm1, 20)
    norm2 = tf.layers.batch_normalization(x2, training=is_train_2)
    norm2 = tf.layers.dropout(norm2, 0.2)
    norm2 = tf.maximum(0.1 * norm2, norm2)

    x3 = tf.layers.dense(norm2, 10)
    norm3 = tf.layers.batch_normalization(x3, training=is_train_2)
    norm3 = tf.layers.dropout(norm3, 0.2)
    norm3 = tf.maximum(0.1 * norm3, norm3)

    x4 = tf.layers.dense(norm3, 4)
    norm4 = tf.layers.batch_normalization(x4, training=is_train_2)
    norm4 = tf.layers.dropout(norm4, 0.2)
    norm4 = tf.maximum(0.1 * norm4, norm4)

    # Output layer logits, fully connected layer with no activation
    logits_2 = tf.layers.dense(norm4, 1, activation=None, name='outputs_2')
    out = tf.tanh(logits_2)
    # Sigmoid cross-entropy loss
    loss_2 = tf.losses.mean_squared_error(labels=targets_2, predictions=out)
    # Mean of the loss
    cost_2 = tf.reduce_mean(loss_2)

    # Adam optimizer
    opt_2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_2)

epochs = 10
batch_size = 1000

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        start = time.time()
        for i in range(0, len(train) - 1000, batch_size):
            samples_0 = np.array([
                np.matmul((j / norm_vals) + 0.0001, embed_mat_w1) + embed_mat_b1
                    for j in train[i:i + batch_size]])

            y = labels[i:i + batch_size]
            y = y.reshape(1000, 1)
            feed = {inputs_2: samples_0,
                    targets_2: y,
                    is_train_2: True}
            train_loss, _ = sess.run([cost_2, opt_2], feed_dict=feed)

            loss += train_loss

            if iteration % 10000 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 10000),
                      "{:.4f} sec/batch".format((end - start) / 10000))
                loss = 0
                start = time.time()

            iteration += 1
    save_path = saver.save(sess, "checkpoints/neural.ckpt")


test_X = []
with open('test.csv', 'r') as file_object:
    line = file_object.readline()
    count = 0
    # skip header
    if line.split(",")[0][0].isalpha():
        line = file_object.readline()
    while line:
        row = line.split(",")
        time_delta = 0

        test_X.append(list(map(int, row[1:6])) + [time_delta])

        line = file_object.readline()
        count += 1

with train_graph.as_default():
    saver = tf.train.Saver()

submission = ['click_id,is_attributed\n']
test_count = 0
with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    for i in range(0, len(test_X), batch_size):
        test_samples = np.array([
                    np.matmul((j / norm_vals) + 0.0001, embed_mat_w1) + embed_mat_b1
                        for j in test_X[i:i + batch_size]])
        predictions = sess.run(out, feed_dict={inputs_2: test_samples, is_train_2: True})
        for k in predictions:
            submission.append('{},{}\n'.format(test_count, 1 if k > 0.05 else 0))
            test_count += 1

with open('/home/ubuntu/talkingdata/submission.csv', 'w') as f:
    f.writelines(submission)
