from flask import Flask, render_template, request
import re
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)

# Read pickle file
pickle_in = open("dataset/plots_text.pickle", "rb")
movie_plots = pickle.load(pickle_in)

# Clean text
movie_plots = [re.sub("[^a-z' ]", "", i) for i in movie_plots]

# Create sequences of length 5 tokens
def create_seq(text, seq_len=5):
    sequences = []
    if len(text.split()) > seq_len:
        for i in range(seq_len, len(text.split())):
            seq = text.split()[i - seq_len : i + 1]
            sequences.append(" ".join(seq))
        return sequences
    else:
        return [text]

seqs = [create_seq(i) for i in movie_plots]
seqs = sum(seqs, [])

# Create inputs and targets (x and y)
x = []
y = []
for s in seqs:
    x.append(" ".join(s.split()[:-1]))
    y.append(" ".join(s.split()[1:]))

# Create integer-to-token mapping
int2token = {}
cnt = 0
for w in set(" ".join(movie_plots).split()):
    int2token[cnt] = w
    cnt += 1

# Create token-to-integer mapping
token2int = {t: i for i, t in int2token.items()}

# Set vocabulary size
vocab_size = len(int2token)

class WordLSTM(nn.Module):
    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_size, 200)

        self.lstm = nn.LSTM(200, n_hidden, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        embedded = self.emb_layer(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        out = self.dropout(lstm_output)
        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            )
        return hidden



#########################################################################33

class WordRNN(nn.Module):

    def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
        super().__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.emb_layer = nn.Embedding(vocab_size, 200)

        ## define the RNN
        # self.lstm = nn.LSTM(200, n_hidden, n_layers,
        #                     dropout=drop_prob, batch_first=True)
        self.rnn = nn.RNN(200, n_hidden, n_layers, dropout=drop_prob, batch_first=True)


        ## define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        ## define the fully-connected layer
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## pass input through embedding layer
        embedded = self.emb_layer(x)

        ## Get the outputs and the new hidden state from the lstm
        rnn_output, hidden = self.rnn(embedded)

        ## pass through a dropout layer
        out = self.dropout(rnn_output)

        #out = out.contiguous().view(-1, self.n_hidden)
        out = out.reshape(-1, self.n_hidden)

        ## put "out" through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden


    def init_hidden(self, batch_size):
        ''' initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        # if GPU is available
        if (torch.cuda.is_available()):
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())

        # if GPU is not available
        else:
          hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

####################################################################


# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose a different model if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text_gpt(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
#######################################################################

@app.route("/")
def home():
    return render_template("index.html")

# Predict next token
def predict(net, tkn, h=None):
    x = np.array([[token2int[tkn]]])
    inputs = torch.from_numpy(x)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)
    p = F.softmax(out, dim=1).data
    p = p.cpu()
    p = p.numpy()
    p = p.reshape(p.shape[1],)
    top_n_idx = p.argsort()[-3:][::-1]
    sampled_token_index = top_n_idx[random.sample([0, 1, 2], 1)[0]]
    return int2token[sampled_token_index], h

# Function to generate text
def sample(net, size, prime="it is"):
    net.eval()
    if torch.cuda.is_available():
        net.cuda()
    h = net.init_hidden(1)
    toks = prime.split()
    for t in prime.split():
        token, h = predict(net, t, h)
        toks.append(token)
    for i in range(size - 1):
        token, h = predict(net, toks[-1], h)
        toks.append(token)
    return " ".join(toks)

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if ((request.method == 'POST') and (request.form['text'] != "") and (request.form['model'] != "")and (request.form['token'] != "")):
      text = request.form['text']
      token = request.form['token']
      model = request.form['model']
      if(model == 'lstm'):
        # Load model
        net = WordLSTM()
        checkpoint = torch.load("model/lstm_nlg.pth", map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model_state_dict"])
        generated_text = sample(net, int(token), prime = text)
      elif(model == 'rnn'):
        # Load model
        net = WordRNN()
        checkpoint = torch.load("model/model_RNN.pth", map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model_state_dict"])
        generated_text = sample(net, int(token), prime = text)
      else:
        generated_text = generate_text_gpt(text, max_length=int(token))
    return generated_text
        

if __name__ == '__main__':
    app.run(debug=True)