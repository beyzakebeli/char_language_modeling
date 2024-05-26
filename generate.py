# importing the necessary packages
import torch
import torch.nn.functional as F
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM

# same one-hot formating function from main.py
def one_hot_encode(sequence, n_chars):
    batch_size, seq_len = sequence.size()
    one_hot = torch.zeros((batch_size, seq_len, n_chars), dtype=torch.float32)
    one_hot.scatter_(2, sequence.unsqueeze(-1), 1.0)
    return one_hot

def generate(model, seed_characters, temperature, idx2char, char2idx, device, seq_length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    model.eval() # model in evaluation mode
    input_seq = torch.tensor([char2idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device) 
    # converting the seed characters into tensors of indices                                                                                                 
    hidden = model.init_hidden(1, device) 
    generated = list(seed_characters)

    for _ in range(seq_length):
        # takes the input sequence and turns into one-hot format
        input_seq_one_hot = one_hot_encode(input_seq, len(char2idx)).to(device) 
        output, hidden = model(input_seq_one_hot, hidden)

        # We can play with the temperature of the Softmax during sampling
        output = output[:, -1, :] /temperature 
        # softmax function to the output logits to get probabilities for each character
        probabilities = F.softmax(output, dim=-1).data.cpu().numpy()
        char_index = np.random.choice(len(char2idx), p=probabilities.ravel())
        
        generated.append(idx2char[char_index])
        
        # preparing the next input
        input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[char_index]], device=device)), dim=1)
    
    samples = ''.join(generated)
        
    return samples



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Shakespeare('shakespeare_train.txt')

    model_path = 'best_model.pth' # getting the best model, CharLSTM

    model = CharLSTM(data.input_size, 128, data.input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # asked chatGPT for the seed characters for a Shakespeare language modeling task
    seed_characters_list = ['To be, or not to be', 'Shall I compare thee', 'O Romeo, Romeo!', 'All the worlds a stage', 'Now is the winter of our discontent']

    temperatures = [0.5, 1.0, 2]  # different temperature values

    # setting the format for the output .txt file
    with open('generated_output.txt', 'w') as f:
        for temp in temperatures:
            f.write(f'Temperature: {temp} \n\n')
            for seed_characters in seed_characters_list:
                f.write(f'Seed: "{seed_characters}"\n')
                generated_text = generate(model, seed_characters, temp, data.idx2char, data.char2idx, device, seq_length=100)
                f.write(f'Generated: "{generated_text}"\n\n')
                f.write('-'*80 + '\n')
            f.write('\n' + '-'*80 + '\n\n')





    