# Character-level Language Model for Shakespeare Data

## Model Implementation

Both RNN and LSTM models have the feature extractor layer and a linear fully connected layer. The feature extractor layer is different for RNN and LSTM.

## Model Performance

### Loss-based performance evaluation
The training and validation loss for the models can be seen in the figure below.
<img width="760" alt="Ekran Resmi 2024-05-26 ÖS 7 19 41" src="https://github.com/beyzakebeli/char_language_modeling/assets/92715108/4e5a1746-6dcc-48aa-8a41-b6b5144157cd">
Both models were trained for 10 epochs using the same network structure and the Adam optimizer with a learning rate of 0.001.  

- **CharRNN Model**:
  - Final Training Loss: 1.4651
  - Final Validation Loss: 1.4669

- **CharLSTM Model**:
  - Final Training Loss: 1.2937
  - Final Validation Loss: 1.2975  

LSTM performs better than vanilla RNN as expected since it is the improved version of the simple RNN architecture. Vanilla RNN structure has a vanishing gradient problem while LSTM's cell state enables it to maintain long-term dependencies more effectively than vanilla RNN. Also, both models show convergence in training and evaluation losses towards the end of the training epochs.  
Based on the loss curves of the models, we can see that LSTM demostrates superior performance over the vanilla RNN model, especially in validation process. 

### Text generation-based performance evalution

For the text generation task, I chose five seed phrases asking ChatGPT. These five phrases were as follows: 'To be, or not to be', 'Shall I compare thee', 'O Romeo, Romeo!', 'All the worlds a stage', 'Now is the winter of our discontent'.

I experimented with three different temperature values: 0.5, 1, and 1.5 to see the effect of temperature in text generation.  

The generated text results can be seen in the generated_output.txt file. To mention here briefly, i will choose the results generated with the 'To be or not to be' phrase.  
For **temperature=0.5**,  
Seed: "To be, or not to be"  
Generated: "To be, or not to be so.  

Second Citizen:  
The Tarruan as thou wast hear me such as you shall: they should be so much the"  

For **temperature=1**,  
Seed: "To be, or not to be"  
Generated: "To be, or not to bear our decessis should  
mineved you mistress'd  
To tail this long not be is have me?  

Messenger:  
Withi"  

For **temperature=1.5**,  
Seed: "To be, or not to be"  
Generated: "To be, or not to being: thouse  
jauld or them, it onh strench him on you Murder follow, To use must deserved soveseif  
Ha"  

The generated text show that the more we increase the temperature, we get more words that don't fit the English language. In the case 
