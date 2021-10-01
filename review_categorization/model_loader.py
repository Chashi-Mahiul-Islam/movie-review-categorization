from torch import nn, optim
import torch

# An Embedding Layer that converts our word_tokens (integers) into embeddings of a speciifc size
# An LSTM Layer defined by a hidden_state size and number of layers
# A fully-connected  output layer that maps the LSTM layer outputs to a desired output_size
# A sigmoid layer which turns all outputs into a value 0-1; return only the last sigmoid output as the output of this network

# init_hidden should initialize the hidden and cell state of an lstm layer to all zeros, and move those state to GPU, if available.

# is GPU available
class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        # initialize the model by setting up the layers

        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
        )

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

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # reurn last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        train_on_gpu = torch.cuda.is_available()
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            )

        return hidden
