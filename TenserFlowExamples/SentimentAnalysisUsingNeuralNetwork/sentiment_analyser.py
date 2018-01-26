import tensorflow as tf
from nltk import word_tokenize
from vectoring_reviews import create_feature_sets_and_labels
import numpy as np

# with open("sentiment_sent.pickle", "r") as f:
#     [X_train, y_train, X_test, y_test] = pickle.load(f)
X_train, y_train, X_test, y_test, lexicons = create_feature_sets_and_labels()

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2

batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_layer_l1 = {"weights": tf.Variable(tf.random_normal([len(X_train[0]), n_nodes_hl1])),
                   "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_layer_l2 = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                   "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_layer_l3 = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                   "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                "biases": tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_layer_l1["weights"]), hidden_layer_l1["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_l2["weights"]), hidden_layer_l2["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_l3["weights"]), hidden_layer_l3["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["biases"])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        hm_epochs = 5
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(X_train):
                start = i
                end = i + batch_size

                epoch_X = X_train[start:end]
                epoch_y = y_train[start:end]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_X, y: epoch_y})
                epoch_loss += c
                i += batch_size

            print("Epoch", epoch, "done out of", hm_epochs, "Loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))

        features = convertToVec("Hey this is kind of works perfectly")
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
        print(prediction.eval(feed_dict={x: [features]}))
        checker(result[0], "Hey this is kind of works perfectly")

        features = convertToVec("He is and idiot  and jerk")
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
        print(prediction.eval(feed_dict={x: [features]}))
        checker(result[0], "He is and idiot  and jerk")


def checker(a, input_data):
    if a == 1:
        print("Negative:", input_data)
    elif a == 0:
        print("negative:", input_data)


def convertToVec(input_data):
    words = word_tokenize(input_data)
    features = np.zeros((len(lexicons)))
    for w in words:
        if w in lexicons:
            index = lexicons.index(w)
            features[index] += 1
    for f in features:
        print(f)
    return features


convertToVec("He is and idiot and jerk")
#train_neural_network(x)
