from string import punctuation


def get_review_with_labels(
    reviews_path="data/reviews.txt", labels_path="data/labels.txt"
):
    """DATA LOADING"""
    # read data from text files
    with open(reviews_path, "r") as f:
        reviews = f.read()
    with open(labels_path, "r") as f:
        labels = f.read()

    return reviews, labels


def remove_punctuations(reviews):
    """DATA PRE-PROCESSING"""
    # get rid of punctuation
    reviews = reviews.lower()  # lowercase, standardize
    all_text = "".join([c for c in reviews if c not in punctuation])

    # split by new lines and and add by spaces
    reviews_split = all_text.split("\n")
    all_text = " ".join(reviews_split)

    return reviews_split, all_text


def text_to_vocab(all_text):
    """CREATING VOCABULARY"""

    # create a list of words
    words = all_text.split()

    return words


def get_train_val_test(features, encoded_labels, split_frac=0.8):
    split_frac = split_frac
    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    # the shapes of resultant feature data
    print("\t\t\tFeature Shapes:")
    print(
        "Train set: \t\t{}".format(train_x.shape),
        "\nValidation set: \t{}".format(val_x.shape),
        "\nTest set: \t\t{}".format(test_x.shape),
    )
    return train_x, train_y, val_x, val_y, test_x, test_y


import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataloader(x, y, batch_size):
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    batch_size = batch_size
    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
    return data_loader
