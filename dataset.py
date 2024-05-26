# importing the necessary packages
import torch
from torch.utils.data import Dataset, DataLoader

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        
        with open(input_file, 'r') as file:
            self.data = file.read() # reading the text file and storing it in self.data

        # duplicate characters are removed 
        # and all the characters in the txt file are sorted alphabetically
        self.chars = sorted(set(self.data))
        self.input_size = len(self.chars) # number of unique characters
        # mapping characters to indices 
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)} 
        # mapping indices to characters
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}

        self.data_indices = [self.char2idx[char] for char in self.data] # characters replaced by their indices

        self.seq_length = 30

    # length of the dataset
    def __len__(self):
        
        length = len(self.data) - self.seq_length
        return length
        

    def __getitem__(self, idx):
        # extracts the sequences of characters 
        input_seq = torch.tensor(self.data_indices[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data_indices[idx + 1:idx + 1 + self.seq_length], dtype=torch.long)
        return input_seq, target_seq

if __name__ == '__main__':

    path = 'shakespeare_train.txt'
    data = Shakespeare(path)
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    # batches of data have this shape: (batch_size, seq_length) 

    for input_seq, target_seq in dataloader:
        print(f"Input sequence: {input_seq}")
        print(f"Target sequence: {target_seq}")
        break  # testing by printing the first input and target sequences
