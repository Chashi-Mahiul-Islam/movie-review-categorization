import numpy as np

''' DATA LOADING '''

# read data from text files 
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()
    
print(reviews[:2000])

print()

print(labels[:20])

''' DATA PRE-PROCESSING '''

from string import punctuation

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and and add by spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

''' CREATING VOCABULARY '''

# create a list of words
words = all_text.split()

print(words[:30])


# ENCODING THE WORDS
from collections import Counter
#counts unique words and track the number of usage
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
# we will do padding with 0 so we will enumerate from 1
vocab2int = {word: index for index, word in enumerate(vocab, 1)}


''' TOKENIZING: WORD TO NUMBER '''

# tokenize review 
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab2int[word] for word in review.split()])

# testing my code   
print('First Tokenized review: \n', reviews_ints[:1])

# ENCODING THE LABELS
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

''' REMOVING OUTLIERS '''

# Getting rid of extremely long or short reviews; the outliers
# Padding/truncating the remaining data so that we have reviews of the same length.

# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Min-length reviews: {}".format(min(review_lens)))
print("Zero-length review: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

print('Number of reviews before removing outliers: ', len(reviews_ints))

# removing zero length review and their labels
non_zero_idx = [ind for ind, review in enumerate(reviews_ints) if len(review) != 0]
reviews_ints = [reviews_ints[i] for i in non_zero_idx]
encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints))

# padding short reviews with 0 till seq_length and truncating long reviews to seq_length

# The review is ['best', 'movie', 'ever'], [117, 18, 128] as integers, the row will look like [0, 0, 0, ..., 0, 117, 18, 128].
# For reviews longer than seq_length, the first seq_length words as the feature vector.

def pad_features(reviews_ints, seq_length):
    
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
        
    return features

# testing padding
seq_length = 200
features = pad_features(reviews_ints, seq_length=seq_length)

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches 
print(features[:30,:10])

''' TRAINING, VALIDATION AND TEST DATASET WITH DATALOADERS '''

split_frac = 0.8
split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]


# the shapes of resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

# DataLoaders and Batching

import torch
from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

#dataLoaders
batch_size = 50

# suffle training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

''' SENTIMENT NETWORK IN PYTORCH '''

# An Embedding Layer that converts our word_tokens (integers) into embeddings of a speciifc size
# An LSTM Layer defined by a hidden_state size and number of layers
# A fully-connected  output layer that maps the LSTM layer outputs to a desired output_size
# A sigmoid layer which turns all outputs into a value 0-1; return only the last sigmoid output as the output of this network

# init_hidden should initialize the hidden and cell state of an lstm layer to all zeros, and move those state to GPU, if available.

# is GPU available
train_on_gpu = torch.cuda.is_available()

if (train_on_gpu):
   print('Training on GPU.')
else: 
    print('No GPU available, training on CPU')

from torch import nn, optim

class SentimentRNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        # initialize the model by setting up the layers 
        
        super(SentimentRNN, self).__init__()
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        
        # dropuout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        # perform a forward pass of our model on some input and hidden state
        batch_size = x.size(0)
        
        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid
        sig_out = self.sig(out)
        
        #reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        #reurn last sigmoid output and hidden state
        return sig_out, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers,
                                 batch_size,
                                 self.hidden_dim).zero_().cuda(),
                        weight.new(self.n_layers,
                               batch_size,
                               self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers,
                                 batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.n_layers,
                                 batch_size,
                                 self.hidden_dim).zero_())
        
        return hidden
        

# vocab_size: Size of our vocabulary or the range of values for our input, word tokens.
# output_size: Size of our desired output; the number of class scores we want to output (pos/neg).
# embedding_dim: Number of columns in the embedding lookup table; size of our embeddings.
# hidden_dim: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
# n_layers: Number of LSTM layers in the network. Typically between 1-3

''' LOADING MODEL, LOSS FUNCTION AND OPTIMIZER '''

# instantiate the model
vocab_size = len(vocab2int) + 1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

# TRAINING


# BCELoss = Binary Cross Entropy Loss for a single Sigmoid output

lr = 0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

''' TRIANING AND VALIDATION '''
valid_loss_min = np.Inf

epochs = 4  # converges in 4
counter = 0
print_every = 100 
clip = 5 # gradient clipping

if(train_on_gpu):
    net.cuda()
    
net.train()



# train for some number of epochs
for e in range(epochs):
    # initialize the hidden state
    h = net.init_hidden(batch_size)
    
    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
            
        # creating new variables for the hidden state, otherwise
        # we'd backdrop through the entire training history
        h = tuple([each.data for each in h])
        
        optimizer.zero_grad()
        
        output, h = net(inputs, h)
        
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in val_loader:
                
                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])
                
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                
                val_losses.append(val_loss.item())
                
                net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...'.format(
                        valid_loss_min,
                        val_loss))
                torch.save(net.state_dict(), 'lstm_model_movie_review_categorization.pt')
                valid_loss_min = np.mean(val_losses)
          
''' TESTING '''

'''
1) Test data performance

2) Inference on user-generated data
'''
# test data performance
net.load_state_dict(torch.load('lstm_model_movie_review_categorization.pt'))

if(train_on_gpu):
    net.cuda()

test_losses = []
num_correct = 0

h = net.init_hidden(batch_size)

net.eval()

for inputs, labels in test_loader:
    
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    h = tuple([each.data for each in h])
    
    output, h = net(inputs, h)
    
    test_loss = criterion(output.squeeze(), labels.float())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    
    num_correct += np.sum(correct)
    
    test_losses.append(test_loss)
# stats
# avg test loss 
print("Test loss: {:.3f}".format(np.mean(test_losses)))
    
# accuracy over all test data
test_acc = num_correct/ len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# inference 

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

from string import punctuation 

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])
    
    # splitting by spaces
    test_words = test_text.split()
    
    # tokens
    test_ints = []
    test_ints.append([vocab2int[word] for word in test_words])
    
    return test_ints

test_ints = tokenize_review(test_review_neg)
print(test_ints)

seq_length = 200 
features = pad_features(test_ints, seq_length)

print(features)

feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())

def predict(net, test_review, sequence_length=200):
    
    net.eval()
    
    test_ints = tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into our model
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
        
    h = net.init_hidden(batch_size)
    
    output, h = net(feature_tensor, h)
    
    # convert output probabilities to predicted class (0, or 1)
    
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
        
    
# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'
test_review_pos = 'best movie ever.'

# negative review 
test_review_neg = 'Not a good movie.'

seq_length = 200
predict(net, test_review_pos, seq_length)