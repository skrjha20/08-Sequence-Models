import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import pandas as pd

def sentence_to_avg(sentence, word_to_vec_map):
    words = sentence.lower().split()
    avg = np.zeros(word_to_vec_map[words[0]].shape)
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    return avg

def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    np.random.seed(1)
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    Y_oh = convert_to_one_hot(Y, C=n_y)

    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples
            avg = sentence_to_avg(X[i], word_to_vec_map)
            z = np.matmul(W, avg) + b
            a = softmax(z)
            cost = -np.sum(Y_oh[i] * np.log(a))
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)
    return pred, W, b

def test_model():
    print(X_train.shape)
    print(Y_train.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(X_train[0])
    print(type(X_train))
    Y = np.asarray([5, 0, 0, 5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
    print(Y.shape)

    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
                    'Lets go party and drinks', 'Congrats on the new job', 'Congratulations',
                    'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
                    'You totally deserve this prize', 'Let us go play football',
                    'Are you down for football this afternoon', 'Work hard play harder',
                    'It is suprising how people can be dumb sometimes',
                    'I am very disappointed', 'It is the best day in my life',
                    'I think I will end up alone', 'My life is so boring', 'Good job',
                    'Great so awesome'])

    print(X.shape)
    print(np.eye(5)[Y_train.reshape(-1)].shape)
    print(type(X_train))


if __name__ == "__main__":
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    maxLen = len(max(X_train, key=len).split())
    index = 1
    print(X_train[index], label_to_emoji(Y_train[index]))

    Y_oh_train = convert_to_one_hot(Y_train, C=5)
    Y_oh_test = convert_to_one_hot(Y_test, C=5)
    index = 50
    print(Y_train[index], "is converted into one hot", Y_oh_train[index])

    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

    word = "cucumber"
    index = 289846
    print("the index of", word, "in the vocabulary is", word_to_index[word])
    print("the", str(index) + "th word in the vocabulary is", index_to_word[index])

    avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
    print("avg = ", avg)

    test_model()
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print(pred)

    print("Training set:")
    pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    print('Test set:')
    pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

    pred = predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    print_predictions(X_my_sentences, pred)

    print(Y_test.shape)
    print('           ' + label_to_emoji(0) + '    ' + label_to_emoji(1) + '    ' + label_to_emoji(
        2) + '    ' + label_to_emoji(3) + '   ' + label_to_emoji(4))
    print(pd.crosstab(Y_test, pred_test.reshape(56, ), rownames=['Actual'], colnames=['Predicted'], margins=True))
    plot_confusion_matrix(Y_test, pred_test)

