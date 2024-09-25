from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import os
import string
import re

#data preprocessing
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_data_pairs(filename):
    data_pairs = []
    converasations = load_doc(filename)
    converasations = converasations.split('\n')
    for pairs in converasations:
        pair = pairs.split(' +++$+++ ')
        data_pairs.append(pair)
    return data_pairs

def vocab_frequency(data_pairs):
    vocab = {}
    for pair in data_pairs:
        for conv_tile in pair:
            for word in conv_tile.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    return vocab

def vocab(vocab_dict):
    threshold = 2
    vocab = []
    for word, val in vocab_dict.items():
        if val > threshold:
            vocab.append(word)
    return vocab

def clean(data_pairs, vocab):
  for pair in range(len(data_pairs)):
    for i in range(len(data_pairs[pair])):
      sentence = data_pairs[pair][i].split(' ')
      for j in range(len(sentence)):
        if sentence[j] not in vocab:
          sentence[j] = ''
      sentence = ' '.join(sentence)
      data_pairs[pair][i] = sentence
  return data_pairs

def tokenize(words, conv_tile):
    vocab = words
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(vocab)

    tensor = tokenizer.texts_to_sequences(conv_tile)
    tensor = pad_sequences(tensor, padding='post')
    return tokenizer, tensor

def dataset_for_training(words, data_pairs):
    questions = []
    replies = []
    for pair in data_pairs:
        questions.append(pair[0])
        replies.append(pair[1])
    input_tokenizer, input_tensor = tokenize(words, questions)
    target_tokenizer, target_tensor = tokenize(words, replies)
    return input_tensor, target_tensor, input_tokenizer, target_tokenizer
                

data_pairs = load_data_pairs('conversations.txt')

vocab_frequency = vocab_frequency(data_pairs)
vocab = vocab(vocab_frequency)
vocab_size = len(vocab) + 1

data_pairs = clean(data_pairs, vocab)

input_tensor, target_tensor, input_tokenizer, target_tokenizer = dataset_for_training(vocab, data_pairs)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.3
)

buffer_size = len(input_tensor_train)
batch_size = 64
steps_per_epoch = len(input_tensor_train)
embedding_dim = 360
units = 1535

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'))

  def call(self, x, hidden): #x is the input
    x = self.embedding(x)
    output, forward_h, backward_h = self.gru(x, initial_state=hidden)
    state = tf.concat([forward_h, backward_h], 1)
    return output, state

  def init_hidden_state(self):
    return [tf.zeros((self.batch_size, self.enc_units)) for i in range(2)]

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values): #query is refered as the hidden states
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    #score = query_with_time_axis(self.W(values))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1) #sum of elements

    return attention_weights, context_vector

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
    super(Decoder, self).__init__()
    self.batch_size = batch_size
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # for bidirectional rnn 
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'))
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, dec_input, dec_hidden, enc_output):
    attention_weights, context_vector = self.attention(dec_hidden, enc_output)
    dec_input = self.embedding(dec_input)
    dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
    #output, state = self.gru(dec_input)
    output, forward_h, backward_h = self.gru(dec_input)
    state = tf.concat([forward_h, backward_h], 1)
    output = tf.reshape(output, (-1, output.shape[2]))
    dec_input = self.fc(output)

    return dec_input, state

encoder = Encoder(vocab_size, embedding_dim, units, batch_size)
decoder = Decoder(vocab_size, embedding_dim, units, batch_size)

#testing
#restore checkpoint
optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = './training_checkpoints_bi'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#clean sentence
def clean_sentence(sentence):
    table = str.maketrans('','',string.punctuation)
    sentence = sentence
    sentence = sentence.lower()
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", sentence)
    words = sentence.split()
    #converts to lowercase
    words = [word.lower() for word in words]
    #remove punctuation from each token
    words = [word.translate(table) for word in words]
    #remove hanging 's and a 
    #remove tokens with numbers in them
    words = [word for word in words if(word.isalpha())]
    sentence = '<start> ' + ' '.join(words) + ' <end>'
    sentence_splitted = sentence.split()
    for i, word in enumerate(sentence_splitted):
      if word not in vocab:
        sentence_splitted[i] = ''
    sentence_splitted = ' '.join(sentence_splitted)
    sentence = sentence_splitted

    return sentence

#test
max_length = 22
def evaluate(sentence):
  sentence = clean_sentence(sentence)
  inputs = [input_tokenizer.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = ''
  hidden = [tf.zeros((1, units)) for i in range(2)]
  enc_out, enc_hidden = encoder(inputs, hidden)
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)
  for t in range(max_length):
    predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += target_tokenizer.index_word[predicted_id] + ' '
    if target_tokenizer.index_word[predicted_id] == '<end>':
      return result, sentence
    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)
    result = re.sub('<end>', '', result)
    return result

print('######CHAT STARTED######')

try: 
    while True:
        user_input = input('Input: ')
        reply = translate(user_input)
        print('Reply: ' + reply)
except KeyboardInterrupt:
    print('chat has stopped')
