import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from .data_structs import Dataset_gen
from .models import NoisyGRUSeq2SeqWithFeatures
from .utils import Variable

def train(data_dir, data_file, save_file, hparams, load_file = None):
  """ Train Network Using Pytorch """
  data_file = ###
  save_file = ####

  ....

  batch_size = hparams.batch_size ....
  ### Other hparams

  data = Dataset_gen(... mol_file, features_file)

  ## Step to get the sequences and sequences length out and features

  loader = DatasetLoader(
    data, 
    batch_size = hparams.batch_size
    shuffle = hparams.
    drop_last = hparams.
    collate_fn = Dataset_gen.collate_function)

  model = NoisyGRUSeq2SeqWithFeatures(..., hparams)

  if load_file:
    model.load_state_dict(torch.load(load_file))

  optimizer = torch.optim.Adam(model.parameters(), lr = hparams.lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = ??? )

  ### TF has a weird gradient clipping step... 

  for epoch in range(hparams.num_train_epochs):
    #for step, (smiles_batch, vec_batch, features_batch) in tqdm( enumerate(loader), total = len(loader)):
    for step, (seq1, seq2, seq1_len, seq2_len, features) in tqdm( enumerate(loader), total = len(loader)):
      
      #if with_features:
      #  seq1, seq2, seq1_len, seq2_len, mol_features = self.iterator.get_next()
      #else:
      #  seq1, seq2, seq1_len, seq2_len = self.iterator.get_next() !!!!!!!
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
      

      ### TF RESHAPE / TF SHAPE
      #shifted_target_len = tf.reshape(target_len, [tf.shape(target_len)[0]]) - 1
      shifted_target_len = torch.reshape(target_len, (torch.size(target_len)[0])) - 1
      ### TF SLICE
      #shifted_target_seq = tf.slice(target_seq, [0, 1], [-1, -1])
      shifted_target_seq = target_seq[0:,1:] ## I think this is correct
      
      ### TF SEQUENCE MASK
      #arget_mask = tf.sequence_mask(shifted_target_len, dtype=tf.float32) ## maxlen = None
      #tf.sequence_mask( [1,2] ) -> [True, False, ...], [True, True, False, ...]#
      ### Produce masks
      target_mask = torch.arange(maxlen)[None, :] < shited_target_len[:, None]

      ### TF REDUCE SUM
      #target_mask = target_mask / tf.reduce_sum(target_mask)
      target_mask = target_mask / torch.sum(target_mask)
      ### TF RESHAPE/ TF SHAPE
      #input_len = tf.reshape(input_len, [tf.shape(input_len)[0]])
      input_len = torch.reshape(input_len, (torch.size(input_len)[0]))
     
      #### CHECK
      encoder_emb_inp, decoder_emb_inp = model._emb_lookup(input_seq, target_seq)
      #if with_features:
      #  return (input_seq, shifted_target_seq, input_len, shifted_target_len,
      #    target_mask, encoder_emb_inp, decoder_emb_inp, mol_features)
      #else:
      #  return (input_seq, shifted_target_seq, input_len, shifted_target_len,
      #    target_mask, encoder_emb_inp, decoder_emb_inp)

      #### OR LOOKING AT MODELS.py
      embedding = model.encode( encoder_emb_inp?, seq_1, seq1_lengths )

      logits = model.decode(   )

      #... = model.liklihood(seq, vec)
      #shifted_target_seq = ???

      loss = F.cross_entropy(logits, self.shifted_target_seq)   ### grab from the file ## Check the order
      loss = torch.sum(loss * self.target_mask)
      loss.backward()
      optimizer.step()

      ### Check perforemance and save a checkpoint of the model
      if( step % hparams.decrease_lr... and step != 0 ):
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
        torch.,save(model.state_dict(), save_file)
   return ...


if __name__ == "__main__":
  train()

