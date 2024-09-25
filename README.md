# Text to Text Chatbot
This Artificial Neural Network (ANN) can generate responses to user inputs, this means that you can chat with it! :O

## Model Architecture
  The model works with an encoder and decoder:
  - ### Encoder

    The encoder processes the input sentence and encodes it into a high-dimensional vector (context vector). This implementation uses a Bidirectional Gated Recurrent Unit (GRU) to capture information from both past and future words in the input sequence.
    Decoder

  - ### Decoder
  
    The decoder generates the response using the context vector from the encoder. It uses the Bahdanau Attention mechanism to focus on relevant parts of the input sentence during the generation of each word in the output sequence. The decoder is also built using a GRU network.

## Dataset
The model was trained on a modified version of the Cornell Movie Dialogs Corpus, which contains over 220,000 conversational exchanges between more than 10,000 pairs of movie characters. This dataset provides a rich source of conversational data suitable for training a conversational AI model.

You can find the modified version of the dataset in the [conversations.txt file](https://github.com/juanmartin8a/Text2Text-Chatbot/blob/main/conversations.txt) from this repo :)

## Model Workflow
  1. ### Data pre-processing
      The conversational data is cleaned and tokenized. During training, words that appear less than a specified threshold are filtered out to reduce the vocabulary size.

  2. ### Inference
      When a user inputs a sentence, the model generates a response by encoding the input and then decoding it to produce the output sentence.


## Disclaimer

The model was trained using the GPUs provided by Google Colab's free tier.
