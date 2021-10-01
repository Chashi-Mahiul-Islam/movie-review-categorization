import torch
import review_categorization.utils as ut


def predict(net, test_review, vocab2int, sequence_length=200):
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        net.cuda()
    net.eval()

    test_ints = ut.tokenize_review(test_review, vocab2int)

    # pad tokenized sequence
    seq_length = sequence_length
    features = ut.pad_features(test_ints, seq_length)

    # convert to tensor to pass into our model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    if train_on_gpu:
        feature_tensor = feature_tensor.cuda()

    h = net.init_hidden(batch_size)

    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0, or 1)

    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print("Prediction value, pre-rounding: {:.6f}".format(output.item()))

    # print custom response
    if pred.item() == 1:
        print("Positive review detected!")
    else:
        print("Negative review detected.")
