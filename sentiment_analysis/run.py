import tensorflow as tf
from tensorflow.python.platform import gfile
import random
import os
import sys
import numpy as np
sys.path.append('../sentiment_analysis/')
import dataset
import dataset

import numpy as np
import random
import tensorflow as tf

  
class discriminator():
  
  

  def __init__(self, vocab_size, unit_size, batch_size, max_length, mode):
    self.vocab_size = vocab_size
    self.unit_size = unit_size
    self.batch_size = batch_size
    self.max_length = max_length
    self.mode = mode

    self.build_model()
    self.saver = tf.train.Saver(max_to_keep = 2)

  def build_model(self):
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size])
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    self.seq_length = tf.placeholder(tf.int32, [None])
    
    _, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    
    w = tf.get_variable('w', [self.unit_size, 1])
    b = tf.get_variable('b', [1])
    output = tf.matmul(hidden_state, w) + b

    self.logit = tf.nn.sigmoid(output)

    if self.mode != 'test':
      self.target = tf.placeholder(tf.float32, [None, 1])
      self.loss = tf.reduce_mean(tf.square(self.target - self.logit))

      self.opt = tf.train.AdamOptimizer().minimize(self.loss)
    else:
      self.vocab_map, _ = dataset.read_map('sentiment_analysis/corpus/mapping')


  def step(self, session, encoder_inputs, seq_length, target = None):
    input_feed = {}
    input_feed[self.encoder_input] = encoder_inputs
    input_feed[self.seq_length] = seq_length

    output_feed = []

    if self.mode == 'train':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      output_feed.append(self.opt)
      #output_feed.append(self.encoder_input)
      #output_feed.append(self.target)
      outputs = session.run(output_feed, input_feed)
      #return outputs[0], outputs[2], outputs[3]
      return outputs[0]
    elif self.mode == 'valid':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]
    elif self.mode == 'test':
      output_feed.append(self.logit)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]

  def get_batch(self, data):
    encoder_inputs = []
    encoder_length = []
    target = []

    for i in range(self.batch_size):
      pair = random.choice(data)
      #pair = data[i]
      length = len(pair[1])
      target.append([pair[0]])
      if length > self.max_length:
        encoder_inputs.append(pair[1][:self.max_length])
        encoder_length.append(self.max_length)
      else:
        encoder_pad = [dataset.PAD_ID] * (self.max_length - length)
        encoder_inputs.append(pair[1] + encoder_pad)
        encoder_length.append(length)

    batch_input = np.array(encoder_inputs, dtype = np.int32)
    batch_length = np.array(encoder_length, dtype = np.int32)
    batch_target = np.array(target, dtype = np.float32)

    return batch_input, batch_length, batch_target

VOCAB_SIZE = 10000
BATCH_SIZE = 32
UNIT_SIZE = 256
MAX_LENGTH = 40
CHECK_STEP = 1000.

def create_model(session, mode):
  m = discriminator(VOCAB_SIZE,
                          UNIT_SIZE,
                          BATCH_SIZE,
                          MAX_LENGTH,
                          mode)
  ckpt = tf.train.get_checkpoint_state('sentiment_analysis/saved_model/')

  if ckpt:
    print("Reading model from %s" % ckpt.model_checkpoint_path)
    m.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters")
    session.run(tf.global_variables_initializer())

  return m

def train():
   if gfile.Exists('corpus/mapping') and gfile.Exists('corpus/SAD.csv.token'):
     print('Files have already been formed!')
   else:
     dataset.form_vocab_mapping(50000)
     vocab_map, _ = dataset.read_map('corpus/mapping')
     dataset.file_to_token('corpus/SAD.csv', vocab_map)

   d = dataset.read_data('corpus/SAD.csv.token')
   random.shuffle(d)    
   
   train_set = d[:int(0.9 * len(d))]
   valid_set = d[int(-0.1 * len(d)):]

   sess = tf.Session()

   Model = create_model(sess, 'train')
   #Model = create_model(sess, 'valid')
   step = 0
   loss = 0

   while(True):
     step += 1
     encoder_input, encoder_length, target = Model.get_batch(train_set)
     '''
     print(encoder_input)
     print(encoder_length)
     print(target)
     exit()
     '''
     loss_train = Model.step(sess, encoder_input, encoder_length, target)
     loss += loss_train/CHECK_STEP
     if step % CHECK_STEP == 0:
       Model.mode = 'valid'
       temp_loss = 0
       for _ in range(100):
         encoder_input, encoder_length, target = Model.get_batch(valid_set)
         loss_valid = Model.step(sess, encoder_input, encoder_length, target)
         temp_loss += loss_valid/100.
       Model.mode = 'train'
       print("Train Loss: %s" % loss)
       print("Valid Loss: %s" % temp_loss)
       checkpoint_path = os.path.join('saved_model/', 'dis.ckpt')
       Model.saver.save(sess, checkpoint_path, global_step = step)
       print("Model Saved!")
       loss = 0

def evaluate():
  vocab_map, _ = dataset.read_map('corpus/mapping')
  sess = tf.Session()
  Model = create_model(sess, 'test')
  Model.batch_size = 1
  
  sys.stdout.write('>')
  sys.stdout.flush()
  sentence = sys.stdin.readline()

  while(sentence):
    token_ids = dataset.convert_to_token(sentence, vocab_map)
    encoder_input, encoder_length, _ = Model.get_batch([(0, token_ids)]) 
    score = Model.step(sess, encoder_input, encoder_length)
    print('Score: ' + str(score[0][0]))
    print('>', end = '')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
if __name__ == '__main__':
  train()
  #evaluate()
