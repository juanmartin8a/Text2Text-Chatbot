import tensorflow as tf
import os
import string
import re
import pickle

# Define hyperparameters
embedding_dim = 360
units = 1535
batch_size = 1  # For inference, batch size is typically 1

# Load the vocabulary and tokenizers
with open('input_tokenizer.pkl', 'rb') as handle:
    input_tokenizer = pickle.load(handle)

with open('target_tokenizer.pkl', 'rb') as handle:
    target_tokenizer = pickle.load(handle)

vocab = set(input_tokenizer.word_index.keys())
vocab_size = len(vocab) + 1

# Define the model
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x, initial_state=hidden)
        state = tf.concat([forward_h, backward_h], 1)
        return output, state

    def init_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)) for _ in range(2)]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, hidden size)
        # values shape: (batch_size, max_len, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return attention_weights, context_vector

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.dec_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, dec_input, dec_hidden, enc_output):
        attention_weights, context_vector = self.attention(dec_hidden, enc_output)
        dec_input = self.embedding(dec_input)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        output, forward_h, backward_h = self.gru(dec_input)
        state = tf.concat([forward_h, backward_h], 1)
        output = tf.reshape(output, (-1, output.shape[2]))
        dec_output = self.fc(output)
        return dec_output, state

encoder = Encoder(vocab_size, embedding_dim, units, batch_size)
decoder = Decoder(vocab_size, embedding_dim, units, batch_size)

# Restore the latest checkpoint for inference
optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = './weights'
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

def clean_sentence(sentence):
    table = str.maketrans('', '', string.punctuation)
    sentence = sentence.lower()
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", sentence)
    words = sentence.split()
    words = [word.translate(table) for word in words]
    words = [word for word in words if word.isalpha()]
    sentence = '<start> ' + ' '.join(words) + ' <end>'
    # Filter out words not in vocabulary
    sentence_splitted = sentence.split()
    sentence_splitted = [word if word in vocab else '' for word in sentence_splitted]
    sentence = ' '.join(sentence_splitted)
    return sentence

max_length = 22

def evaluate(sentence):
    sentence = clean_sentence(sentence)
    inputs = [input_tokenizer.word_index.get(word, 0) for word in sentence.split()]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = encoder.init_hidden_state()
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    for _ in range(max_length):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()

        if predicted_id == 0:
            break

        predicted_word = target_tokenizer.index_word.get(predicted_id, '')
        if predicted_word == '<end>':
            break

        result += predicted_word + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

def translate(sentence):
    result = evaluate(sentence)
    return result

print('###### CHAT STARTED ######')

try:
    while True:
        user_input = input('Input: ')
        if user_input.lower() in ['/exit', '/quit', '/bye']:
            print('Chat has stopped.')
            break
        reply = translate(user_input)
        print('Reply: ' + reply)
except KeyboardInterrupt:
    print('\nChat has stopped.')
