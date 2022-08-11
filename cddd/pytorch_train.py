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

  train_model = NoisyGRUSeq2SeqWithFeatures(..., hparams)

  if load_file:
    train_model.load_state_dict(torch.load(load_file))

  optimizer = torch.optim.Adam(model.parameters(), lr = hparams.lr)

  for epoch in range(hparams.num_train_epochs):
    for step, (smiles_batch, vec_batch, features_batch) in tqdm( enumerate(loader), total = len(loader)):
      .....

      ... = model.liklihood(seq, vec)
      loss = F.crossent()   ### grab from the file 
      loss.backward()
      optimizer.step()

      ### Check perforemance and save a checkpoint of the model
      if( step % hparams.decrease_lr... and step != 0 ):
        decrease_learning_rate(optimizer, decrease_by = ... torch.exponentiallrscehdulaer?)
        tqdm.write("#" * 49)
        tqdm.write("Ep. {:4d}, step {:4d}, loss {:5.3f}\n".format(epoch, step, loss.data.item()))
        x, y, z = model.sample(batch_size, vecs) ### ????
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

