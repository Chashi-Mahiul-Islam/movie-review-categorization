import torch
import review_categorization.data_loader as dl
import review_categorization.utils as ut
import review_categorization.model_loader as ml
import review_categorization.trainer as tr
import review_categorization.tester as ts
import review_categorization.predictor as p


def categorize(rev_path, label_path):
    """DATA LOADING"""
    reviews, labels = dl.get_review_with_labels(rev_path, label_path)

    print(reviews[:2000])

    print()

    print(labels[:20])

    """ DATA PRE-PROCESSING """

    reviews_split, all_text = dl.remove_punctuations(reviews)

    """ CREATING VOCABULARY """

    # create a list of words
    words = dl.text_to_vocab(all_text)

    print(words[:30])

    """ ENCODING THE WORDS """
    vocab2int = ut.endcode_words(words)

    """ TOKENIZING: WORD TO NUMBER """

    reviews_ints = ut.tokenize(reviews_split, vocab2int)

    # testing my code
    print("First Tokenized review: \n", reviews_ints[:1])

    """ ENCODING THE LABELS"""
    encoded_labels = ut.encode_labels(labels)

    """ REMOVING OUTLIERS """

    reviews_ints, encoded_labels = ut.remove_outliers(reviews_ints, encoded_labels)

    # testing padding
    seq_length = 200
    features = ut.pad_features(reviews_ints, seq_length=seq_length)

    ## test statements - do not change - ##
    assert len(features) == len(
        reviews_ints
    ), "Your features should have as many rows as reviews."
    assert (
        len(features[0]) == seq_length
    ), "Each feature row should contain seq_length values."

    # print first 10 values of the first 30 batches
    print(features[:30, :10])

    """ TRAINING, VALIDATION AND TEST DATASET WITH DATALOADERS """

    train_x, train_y, val_x, val_y, test_x, test_y = dl.get_train_val_test(
        features, encoded_labels
    )
    # DataLoaders and Batching

    # dataLoaders
    batch_size = 50

    # suffle training data
    train_loader = dl.get_dataloader(train_x, train_y, batch_size)
    val_loader = dl.get_dataloader(val_x, val_y, batch_size)
    test_loader = dl.get_dataloader(test_x, test_y, batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print("Sample input size: ", sample_x.size())  # batch_size, seq_length
    print("Sample input: \n", sample_x)
    print()
    print("Sample label size: ", sample_y.size())  # batch_size
    print("Sample label: \n", sample_y)

    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU")

    # vocab_size: Size of our vocabulary or the range of values for our input, word tokens.
    # output_size: Size of our desired output; the number of class scores we want to output (pos/neg).
    # embedding_dim: Number of columns in the embedding lookup table; size of our embeddings.
    # hidden_dim: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
    # n_layers: Number of LSTM layers in the network. Typically between 1-3

    """ LOADING MODEL, LOSS FUNCTION AND OPTIMIZER """

    # instantiate the model
    vocab_size = len(vocab2int) + 1  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = ml.SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    print(net)

    """ TRAINING """

    # BCELoss = Binary Cross Entropy Loss for a single Sigmoid output

    tr.train(net, train_loader, val_loader, batch_size)

    """ TESTING """

    """
    1) Test data performance
    
    2) Inference on user-generated data
    """
    # test data performance
    net.load_state_dict(torch.load("lstm_model_movie_review_categorization.pt"))
    ts.test(net, test_loader, batch_size)

    """ INFERENCE """

    test_review_neg = "The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow."

    test_ints = ut.tokenize_review(test_review_neg, vocab2int)
    print(test_ints)

    seq_length = 200
    features = ut.pad_features(test_ints, seq_length)

    print(features)

    feature_tensor = torch.from_numpy(features)
    print(feature_tensor.size())

    # positive test review
    test_review_pos = (
        "This movie had the best acting and the dialogue was so good. I loved it."
    )
    test_review_pos = "best movie ever."

    # negative review
    test_review_neg = "Not a good movie."

    seq_length = 200
    p.predict(net, test_review_pos, vocab2int, seq_length)
