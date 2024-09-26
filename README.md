# Text to Text Chatbot
This Artificial Neural Network (ANN) can generate responses to user inputs. You can chat with it! :O
## Model
  The model works with an encoder-decoder architecture:
  
  - ### Encoder
    The encoder processes the input sentence and encodes it into a high-dimensional vector (context vector). This implementation uses a Bidirectional Gated Recurrent Unit (GRU) to capture information from both past and future words in the input sequence.

  - ### Decoder
    The decoder generates the response using the context vector from the encoder. It uses the Bahdanau Attention mechanism to focus on relevant parts of the input sentence during the generation of each word in the output sequence. The decoder is also built using a Bidirectional GRU network.

The model was trained using the GPUs provided by Google Colab's free tier. This significantly limited the amount of time and compute power.

## Dataset
The model was trained on a modified version of the Cornell Movie Dialogs Corpus, which contains over 220,000 conversational exchanges between more than 10,000 pairs of movie characters. This dataset provides a rich source of conversational data suitable for training a conversational AI model.

The dataset was modified to reduce its size and make training more simple.

You can find the modified version of the dataset in the [conversations.txt file](https://github.com/juanmartin8a/Text2Text-Chatbot/blob/main/conversations.txt) from this repo :)

## Model Workflow
  1. ### Data Pre-processing
      The conversational data is cleaned and tokenized. During training, words that appear less than a specified threshold are filtered out to reduce the vocabulary size.

  2. ### Inference
      When a user inputs a sentence, the model generates a response by encoding the input and then decoding it to produce the output sentence.

## How to use

  1. Clone this repository:
     ```bash
     git clone https://github.com/juanmartin8a/Text2Text-Chatbot.git
     cd Text2Text-Chatbot

  2. Make sure to add the weights to the "weights" directory. To add the weights:
     
      1. Download the [checkpoint](https://drive.google.com/file/d/1LyOLic348znugoMWDB4GZEn-LuZaAxLE/view?usp=sharing), [ckpt-13.index](https://drive.google.com/file/d/1s_cbZErw3PXg45WXMn6Ep4cv0yJhV2SM/view?usp=sharing), and [ckpt-13.data-00000-of-00001](https://drive.google.com/file/d/1lUQ-1gP66HeikvwB9f9eqPSybdQ-pOhq/view?usp=sharing) files. These files contain the weights.
     
      2. Add the files to the "weights" directory :)

  4. Run the program: `python test.py`.

## Disclaimer
There are a few things to note about this model:
  - Because of the way the model was trained to save resources, the model can only read the current user input to generate a response so it doesn't have memory of previous user inputs :/ . This could be improved by training the model with the complete movie characters dialog instead of only an input and response per dialog.
  - The model could also improve by having a better vocabulary (more tokens).
