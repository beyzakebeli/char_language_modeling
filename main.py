# importing the necessary packages
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM

# CharRNN and CharLSTM take one-hot encoded input sequences
# making sure the input sequences in the training process are also in one-hot encoded format
def one_hot_encode(sequence, n_chars):
    batch_size, seq_len = sequence.size()
    one_hot = torch.zeros((batch_size, seq_len, n_chars), dtype=torch.float32)
    one_hot.scatter_(2, sequence.unsqueeze(-1), 1.0)
    return one_hot

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train() # model in training mode
    total_loss = 0.0 # initializing loss
    for input_seq, target_seq in trn_loader:
        # moving the sequences to the device
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        # coverting the input sequence to one-hot encoded format and sending to device
        input_seq = one_hot_encode(input_seq, model.input_size).to(device)
        # initializing the hidden state
        hidden = model.init_hidden(input_seq.size(0), device)  
        optimizer.zero_grad() # to prevent gradient accumulation
        output, hidden = model(input_seq, hidden) # performing forward pass
        loss = criterion(output.view(-1, model.output_size), target_seq.view(-1)) 
        # loss values calculated
        loss.backward() # backpropagation to get the gradients of loss
        optimizer.step()
        total_loss += loss.item()

    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval() # model in evaluation mode
    total_loss = 0.0 # initializing loss
    with torch.no_grad(): # disabling gradient calculation for fast computation
        for input_seq, target_seq in val_loader:
            # moving the sequences to the device
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            # coverting the input sequence to one-hot encoded format and sending to device
            input_seq = one_hot_encode(input_seq, model.input_size).to(device)
            # initializing the hidden state
            hidden = model.init_hidden(input_seq.size(0), device)  
            output, hidden = model(input_seq, hidden) # performing forward pass
            # loss values calculated
            loss = criterion(output.view(-1, model.output_size), target_seq.view(-1))
            total_loss += loss.item()
            
    val_loss = total_loss / len(val_loader)
    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Shakespeare('shakespeare_train.txt')


    # Splitting for training and validation
    indices = list(range(len(data)))
    split = int(0.8 * len(data))  # 80% train, 20% validation
    np.random.shuffle(indices)  # shuffle indices randomly

    train_indices, val_indices = indices[:split], indices[split:]

    # Using SumbsetRandomSampler like requested
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # creating dataloaders for train and validation data
    trn_loader = DataLoader(data, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=64, sampler=val_sampler)

    # initializing the models
    models = {
        "CharRNN": CharRNN(input_size=data.input_size, hidden_size=128, output_size=data.input_size).to(device),
        "CharLSTM": CharLSTM(input_size=data.input_size, hidden_size=128, output_size=data.input_size).to(device)
    }

    # cost function is CrossEntropyLoss as requested
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with 0.001 learning rate
    optimizer = {
        "CharRNN": optim.Adam(models["CharRNN"].parameters(), lr=0.001),
        "CharLSTM": optim.Adam(models["CharLSTM"].parameters(), lr=0.001)
    }

    num_epochs = 10
    # to keep track of the best validation loss and model
    best_val_loss = {"CharRNN": float('inf'), "CharLSTM": float('inf')}
    best_model = {"CharRNN": None, "CharLSTM": None}

    history = {
        "CharRNN": {"train_loss": [], "val_loss": []},
        "CharLSTM": {"train_loss": [], "val_loss": []}
    }
    
    for model_name, model in models.items():
        print(f"Training {model_name}")
        for epoch in range(num_epochs):
            trn_loss = train(model, trn_loader, device, criterion, optimizer[model_name])
            val_loss = validate(model, val_loader, device, criterion)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {trn_loss:.4f}, Val Loss:{val_loss:.4f}')
            # saving the loss values in history
            history[model_name]["train_loss"].append(trn_loss)
            history[model_name]["val_loss"].append(val_loss)

            # best validation loss is found
            if val_loss < best_val_loss[model_name]:
                best_val_loss[model_name] = val_loss
                best_model[model_name] = model.state_dict()

    # determine the best model from RNN and LSTM based on validation loss
    best_model_name = min(best_val_loss, key=best_val_loss.get)
    best_model_state_dict = best_model[best_model_name]

    # saving the best model
    torch.save(best_model_state_dict, f'best_model.pth')
    print(f"Saved best model: {best_model_name} with validation loss {best_val_loss[best_model_name]:.4f}")
    
if __name__ == '__main__': 
    main()

""" Output of the main.py file:
Training CharRNN
Epoch [1/10], Train Loss: 2.0067, Val Loss:1.7362
Epoch [2/10], Train Loss: 1.6687, Val Loss:1.6185
Epoch [3/10], Train Loss: 1.5896, Val Loss:1.5670
Epoch [4/10], Train Loss: 1.5484, Val Loss:1.5328
Epoch [5/10], Train Loss: 1.5222, Val Loss:1.5156
Epoch [6/10], Train Loss: 1.5043, Val Loss:1.5021
Epoch [7/10], Train Loss: 1.4908, Val Loss:1.4864
Epoch [8/10], Train Loss: 1.4804, Val Loss:1.4792
Epoch [9/10], Train Loss: 1.4721, Val Loss:1.4723
Epoch [10/10], Train Loss: 1.4651, Val Loss:1.4669
Training CharLSTM
Epoch [1/10], Train Loss: 2.0939, Val Loss:1.7709
Epoch [2/10], Train Loss: 1.6692, Val Loss:1.5914
Epoch [3/10], Train Loss: 1.5402, Val Loss:1.4997
Epoch [4/10], Train Loss: 1.4660, Val Loss:1.4432
Epoch [5/10], Train Loss: 1.4164, Val Loss:1.4031
Epoch [6/10], Train Loss: 1.3803, Val Loss:1.3711
Epoch [7/10], Train Loss: 1.3519, Val Loss:1.3459
Epoch [8/10], Train Loss: 1.3288, Val Loss:1.3278
Epoch [9/10], Train Loss: 1.3097, Val Loss:1.3101
Epoch [10/10], Train Loss: 1.2937, Val Loss:1.2975
Saved best model: CharLSTM with validation loss 1.2975
 """