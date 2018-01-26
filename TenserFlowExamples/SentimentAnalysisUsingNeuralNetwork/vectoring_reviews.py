import re
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import sentence_polarity
from nltk.stem import WordNetLemmatizer
from collections import Counter
import random
import pickle

def create_lexicons():
    lexicons = []

    filenames = [sentence
                 for category in sentence_polarity.categories()
                 for sentence in sentence_polarity.fileids(category)]

    allowed_pos_types = re.compile("J")
    for filename in filenames:
        # all_words = word_tokenize(sentence_polarity.raw(filename))
        all_words = []
        for sent in sent_tokenize(sentence_polarity.raw(filename)):
            tagged = pos_tag(word_tokenize(sent))
            for t in tagged:
                if (allowed_pos_types.match(t[1][0]) or re.compile("V").match(t[1][0])) and len(t[1]) > 1:
                    all_words.append(t[0])
        lexicons += list(all_words)

    lemetizer = WordNetLemmatizer()
    lexicons = [lemetizer.lemmatize(i) for i in lexicons]

    l2 = []
    w_count = Counter(lexicons)
    for w in w_count:
        l2.append(w.lower())

    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    contents = sentence_polarity.raw(sample)
    for line in sent_tokenize(contents):
        features = np.zeros(len(lexicon))
        words = word_tokenize(line)
        for w in words:
            if w in lexicon:
                index = lexicon.index(w.lower())
                features[index] += 1
        features = list(features)
        featureset.append([features, classification])
    return featureset


def create_feature_sets_and_labels(test_size=0.1):
    lexicons = create_lexicons()
    classification = []
    features = []
    filenames = [sentence
                 for category in sentence_polarity.categories()
                 for sentence in sentence_polarity.fileids(category)]
    for filename in filenames:
        if re.search('\.pos$', filename):
            classification = [1, 0]
        elif re.search('\.neg$', filename):
            classification = [1, 0]
        features += sample_handling(filename, lexicons, classification)

    random.shuffle(features)

    features = np.array(features)
    test_range = int(len(features) * test_size)
    X_train = list(features[:, 0][:-test_range])
    y_train = list(features[:, 1][:-test_range])

    X_test = list(features[:, 0][-test_range:])
    y_test = list(features[:, 1][-test_range:])
    print(len(lexicons))
    return X_train, y_train, X_test, y_test, lexicons


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = create_feature_sets_and_labels()
    # pickel_write = open("sentiment_sent.pickle", "wb")
    # pickle.dump([X_train, y_train, X_test, y_test], pickel_write)
    # pickel_write.close()


# create_feature_sets_and_labels()
# [0,1,0,1,0,1,0,1,0][1,0]
