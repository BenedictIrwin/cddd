import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from .data_structs import DatasetWithFeatures, Vocabulary
from .models import NoisyGRUSeq2SeqWithFeatures
from .utils import Variable

def train(hparams):
  """ Train Network Using Pytorch """
  data_file = "./guacamol_v1_train_subset.smiles"
  save_file = "test_save.sav"
  #load_file = hparams.load_file#(else None)
  ## fix this to get everytihng from hparams

  batch_size = hparams.batch_size ## 512 ?
  #data = DatasetWithFeatures(data_file, features_file)


  model = NoisyGRUSeq2SeqWithFeatures("TRAIN", None, hparams)
  vocab = Vocabulary(model.decode_vocabulary)   ### A bit like the Pipeline really
  
  #print(model.decode_vocabulary)
  #for i in model.decode_vocabulary.keys():
  #  print(i)
  #vocab.add_characters(list(model.decode_vocabulary.keys()))
  #print(vocab.__dict__)

  #exit()
  ### Possibly pass hparams to dataset as well!
  data = DatasetWithFeatures(vocab, data_file)

  ## Step to get the sequences and sequences length out and features
  #loader = DatasetLoader(
  #  data, 
  #  batch_size = hparams.batch_size
  #  shuffle = hparams.
  #  drop_last = hparams.
  #  collate_fn = Dataset_gen.collate_function)


  load_file = None
  if load_file:
    model.load_state_dict(torch.load(load_file))

  print(hparams)

  optimizer = torch.optim.Adam(model.parameters(), lr = hparams.lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = hparams.lr_decay_frequency)
  print("VERIFY^ THE CORRECTNESS OF gamma = ... in above!")

  ### TF has a weird gradient clipping step... 

  for epoch in range(hparams.num_steps):
    #for step, (smiles_batch, vec_batch, features_batch) in tqdm( enumerate(loader), total = len(loader)):
    #for step, (seq1, seq2, seq1_len, seq2_len, features) in tqdm( enumerate(loader), total = len(loader)):
    for step, (seq1, seq1_len) in tqdm( enumerate(data), total = len(data)):

      print(step, seq1, seq1_len)
      #### We need to make this work for batches of compounds?


      #if with_features:
      #  seq1, seq2, seq1_len, seq2_len, mol_features = self.iterator.get_next()
      #else:
      #  seq1, seq2, seq1_len, seq2_len = self.iterator.get_next() !!!!!!!
      '''
      if self.rand_input_swap:
        ### TF RANDOM UNIFORM --> MIGHT BE BETTER REPHRASED with a not?
        #rand_val = tf.random_uniform([], dtype=tf.float32)
        rand_val = torch.rand([], dtype = torch.float32)
        ### TF COND
        #input_seq = tf.cond(tf.greater_equal(rand_val, 0.5),
        #                    lambda: seq1, lambda: seq2)
        input_seq = torch.where(rand_val >= 0.5, seq1, seq2)
        ### TF COND
        #input_len = tf.cond(tf.greater_equal(rand_val, 0.5),
        #                    lambda: seq1_len, lambda: seq2_len)
        input_len = torch.where(rand_val >= 0.5, seq1_len, seq2_len)
      else:
        input_seq = seq1
        input_len = seq1_len
        target_seq = seq2
        target_len = seq2_len
      '''
      input_seq = torch.tensor(seq1)
      input_len = torch.tensor(seq1_len)
      target_seq = torch.tensor(seq1)
      target_len = torch.tensor(seq1_len)

      print(target_len.size())

      ### TF RESHAPE / TF SHAPE
      #shifted_target_len = tf.reshape(target_len, [tf.shape(target_len)[0]]) - 1
      shifted_target_len = target_len - 1
      print(shifted_target_len)
      ### TF SLICE
      #shifted_target_seq = tf.slice(target_seq, [0, 1], [-1, -1])
      shifted_target_seq = target_seq[0:,1:] ## I think this is correct
      print(target_seq)
      print(target_seq.size())
      print(shifted_target_seq)
      print(shifted_target_seq.size())

      
      ### TF SEQUENCE MASK
      #arget_mask = tf.sequence_mask(shifted_target_len, dtype=tf.float32) ## maxlen = None
      #tf.sequence_mask( [1,2] ) -> [True, False, ...], [True, True, False, ...]#
      ### Produce masks
      target_mask = torch.arange(target_seq.size()[1])[None, :] < shifted_target_len[:, None]
      

      ### TF REDUCE SUM
      #target_mask = target_mask / tf.reduce_sum(target_mask)
      target_mask = target_mask / torch.sum(target_mask)
      #print(target_mask)
      #input()
      ### TF RESHAPE/ TF SHAPE
      #input_len = tf.reshape(input_len, [tf.shape(input_len)[0]])
      
      #input_len = torch.reshape(input_len, (torch.size(input_len)[0]))
     
      #input("pre emb...")

      #### CHECK
      #encoder_emb_inp, decoder_emb_inp = model._emb_lookup(input_seq, target_seq)
      

      #if with_features:
      #  return (input_seq, shifted_target_seq, input_len, shifted_target_len,
      #    target_mask, encoder_emb_inp, decoder_emb_inp, mol_features)
      #else:
      #  return (input_seq, shifted_target_seq, input_len, shifted_target_len,
      #    target_mask, encoder_emb_inp, decoder_emb_inp)

      #### OR LOOKING AT MODELS.py
      #embedding = model.encode( encoder_input, encoder_lengths )
      embedding = model.encode( input_seq, input_len )



      print(embedding)
      print(embedding.shape)


      #input("post")
      logits = model.decode(embedding)
     
      print("logits",logits) #### Would have thought, vectors such that i.e. softmax would give probabilities over tokens?
      print("shifted,seq",shifted_target_seq)

      input("CHECK THE OUTPUT LOGITS")

      ### Some examples
      #test_inputs = np.load("test_in_seq.npy")
      #tiest_input_len = np.load("test_in_len.npy")
      #test_outputs = np.load("test_output_embeddings.npy")

      ### Call the model
      #encoder = MiniCDDDInference()
      #outputs = encoder(test_inputs, test_input_len)

      #print(outputs)
      #print(outputs.shape)
      #print(test_outputs)


      #decoder = MiniCDDDDecoder()
      #outputs = decoder(outputs)  ## Input N x 512 vectors... generate compounds



      #... = model.liklihood(seq, vec)
      #shifted_target_seq = ???

      loss = F.cross_entropy(logits, shifted_target_seq)   ### grab from the file ## Check the order
      loss = torch.sum(loss * self.target_mask)
      loss.backward()
      optimizer.step()

      ### Check perforemance and save a checkpoint of the model
      if( step % hparams.decrease_lr and step != 0 ):
        #decrease_learning_rate(optimizer, decrease_by = ... torch.exponentiallrscehdulaer?)
        tqdm.write("#" * 49)
        tqdm.write("Ep. {:4d}, step {:4d}, loss {:5.3f}\n".format(epoch, step, loss.data.item()))
        #x, y, z = model.sample(batch_size, vecs) ### ????
        ### Get num valid
        ###
        ###
        ##
        ####
        tqdm.write("#"* 49)
        torch.save(model.state_dict(), save_file)
  return None


if __name__ == "__main__":
  train()

