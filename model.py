# importing the necessary packages
import torch
import torch.nn as nn

# defining vanilla RNN model
class CharRNN(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__() # initializing the base class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # creating the RNN layer (input states to hidden states)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # fully connected layer (hidden states to predicted outputs)
        self.fc = nn.Linear(hidden_size, output_size)

    # forward pass method
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.fc(output) # fully connected layer
        return output, hidden 

    def init_hidden(self, batch_size, device):
        # creating the initial hidden tensor with 0s
        initial_hidden = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return initial_hidden

# defining LSTM model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__() # initializing the base class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # creating the LSTM layer (input states to hidden states)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # fully connected layer (hidden states to predicted outputs)
        self.fc = nn.Linear(hidden_size, output_size)

    # forward pass method
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.fc(output) # fully connected layer
        return output, hidden

    def init_hidden(self, batch_size, device):
        # LSTM has two hidden states, cell state + hidden state
        initial_hidden = (torch.zeros(1, batch_size, self.hidden_size).to(device), 
                          torch.zeros(1, batch_size, self.hidden_size).to(device))
        return initial_hidden
