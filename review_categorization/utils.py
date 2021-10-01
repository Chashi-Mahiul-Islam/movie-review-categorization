import numpy as np
from collections import Counter
from string import punctuation


def endcode_words(words):
    # ENCODING THE WORDS

    # counts unique words and track the number of usage
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    # we will do padding with 0 so we will enumerate from 1
    vocab2int = {word: index for index, word in enumerate(vocab, 1)}

    return vocab2int


def tokenize(reviews_split, vocab2int):
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab2int[word] for word in review.split()])
    return reviews_ints


def tokenize_review(test_review, vocab2int):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = "".join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab2int[word] for word in test_words])

    return test_ints


def encode_labels(labels):
    # ENCODING THE LABELS
    labels_split = labels.split("\n")
    encoded_labels = np.array(
        [1 if label == "positive" else 0 for label in labels_split]
    )
    return encoded_labels


def remove_outliers(reviews_ints, encoded_labels):
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Min-length reviews: {}".format(min(review_lens)))
    print("Zero-length review: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    print("Number of reviews before removing outliers: ", len(reviews_ints))

    # removing zero length review and their labels
    non_zero_idx = [ind for ind, review in enumerate(reviews_ints) if len(review) != 0]
    reviews_ints = [reviews_ints[i] for i in non_zero_idx]
    encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])

    print("Number of reviews after removing outliers: ", len(reviews_ints))
    return reviews_ints, encoded_labels


def pad_features(reviews_ints, seq_length):

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        features[i, -len(row) :] = np.array(row)[:seq_length]

    return features
