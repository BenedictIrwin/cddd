import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cddd.utils import Variable
from cddd.data_structs import Vocabulary

### TRAINING
#import torch
#from torch.utils.data import DataLoader
#import torch.nn.functional as F

from tqdm import tqdm

#from cddd.data_structs import DatasetWithFeatures, Vocabulary
#from cddd.models import NoisyGRUSeq2SeqWithFeatures
#from cddd.utils import Variable


class MultiGRU(nn.Module):
  """Stacked version of MultiGRU with TF like weight structure"""
  def __init__(self, voc_size, latent_vec_sizes):
    super(MultiGRU, self).__init__()
    self.latent_vec_sizes = latent_vec_sizes
    siz = [voc_size] + latent_vec_sizes
    self.grus = nn.ModuleDict({ "gru_{}".format(i) : TFGRUCell(siz[i],siz[i+1]) for i in range(len(latent_vec_sizes))})
    #self.grus = nn.ModuleList([ TFGRUCell(siz[i],siz[i+1]) for i in range(len(latent_vec_sizes))])

     
    #self.gru_0 = TFGRUCell(voc_size, latent_vec_sizes[0])
    #self.gru_1 = TFGRUCell(latent_vec_sizes[0], latent_vec_sizes[1])
    #self.gru_2 = TFGRUCell(latent_vec_sizes[1], latent_vec_sizes[2])
  
  def forward(self, x, h):
    """Run the three GRU cells with zero input"""
    #h_new = [0,0,0]
    h_new = [0 for i in self.latent_vec_sizes]
    for i in range(len(self.latent_vec_sizes)):
      x = h_new[i] = self.grus["gru_{}".format(i)](x, h[i])
    #x = h_new[0] = self.gru_0(x, h[0])
    #x = h_new[1] = self.gru_1(x, h[1])
    #x = h_new[2] = self.gru_2(x, h[2])
    return x, h_new

  def from_pretrained(self, pretrained):
    """Loads up from gate and candidate kernels and biases K_g, K_c, b_g, b_c"""
    for i in range(len(self.latent_vec_sizes)):
      self.grus["gru_{}".format(i)].from_pretrained(pretrained["gru_{}".format(i)])
    #self.gru_0.from_pretrained(pretrained["gru_0"])
    #self.gru_1.from_pretrained(pretrained["gru_1"])
    #self.gru_2.from_pretrained(pretrained["gru_2"])

class TFGRUCell(nn.Module):
  """A GRU Cell made to work exactly as the TF version"""
  def __init__(self, in_size, out_size):
    super().__init__()
    self._activation = torch.tanh
    self.in_size = in_size
    self.out_size = out_size
    
    with torch.no_grad():
      self._gate_kernel      = nn.Parameter(torch.from_numpy(np.random.rand(self.in_size + self.out_size, 2* self.out_size)).float())
      self._gate_bias        = nn.Parameter(torch.from_numpy(np.random.rand(2 * self.out_size)).float())
      self._candidate_kernel = nn.Parameter(torch.from_numpy(np.random.rand(self.in_size + self.out_size, self.out_size)).float())
      self._candidate_bias   = nn.Parameter(torch.from_numpy(np.random.rand(self.out_size)).float())
    #torch.Size([544, 1024])
    #torch.Size([1024])
    #torch.Size([544, 512])
    #torch.Size([512])
    #32
    #512

  def from_pretrained(self, dictionary):
    """ Split up the K_g, K_c, b_g, b_c weights correctly """
    with torch.no_grad():
      self._gate_kernel      = nn.Parameter(torch.from_numpy(dictionary["gates-kernel"]).float())
      self._gate_bias        = nn.Parameter(torch.from_numpy(dictionary["gates-bias"]).float())
      self._candidate_kernel = nn.Parameter(torch.from_numpy(dictionary["candidate-kernel"]).float())
      self._candidate_bias   = nn.Parameter(torch.from_numpy(dictionary["candidate-bias"]).float())
    #print(self._gate_kernel.size()) 
    #print(self._gate_bias.size()) 
    #print(self._candidate_kernel.size())
    #print(self._candidate_bias.size())
    #print(self.in_size)
    #print(self.out_size)
    #exit()

  def forward(self, x, h):
    """ Calculate GRU logic using block matrices """
    gate_inputs = torch.matmul(torch.cat([x, h]), self._gate_kernel) + self._gate_bias
    value = torch.sigmoid(gate_inputs)
    r, z = value[:self.out_size], value[self.out_size:]
    r_state = r * h
    candidate = torch.matmul(torch.cat([x, r_state]), self._candidate_kernel) + self._candidate_bias
    return z * h + (1 - z) * self._activation(candidate)

class NoisyGRUSeq2SeqWithFeatures(nn.Module):
  """NoisyGRUSeq2SeqWithFeatures"""
  
  def __init__(self, hparams):
    super().__init__()

    self.hparams = hparams
    
    self.decode_vocabulary = { v: k for k, v in np.load(self.hparams.decode_vocabulary_file, allow_pickle=True).item().items() }
    self.decode_vocabulary_inv = { k: v for k, v in np.load(self.hparams.decode_vocabulary_file, allow_pickle=True).item().items() }
    self.vocab = Vocabulary(self.decode_vocabulary)
  

    # Load pretrained weights
    #pretrained_encoder_dict = {}
    #for i in ["0","1","2"]:
    #  pretrained_encoder_dict["gru_{}".format(i)] = {}
    #  for j in ["candidate", "gates"]:
    #    for k in ["bias","kernel"]:
    #      file_name = "Encoder/rnn/multi_rnn_cell/cell_{}/gru_cell/{}/{}.npy".format(i,j,k).replace("/","-")
    #      pretrained_encoder_dict["gru_{}".format(i)]["{}-{}".format(j,k)] = np.load(file_name)

    self.encoder_embedding = nn.Embedding(self.hparams.voc_size, self.hparams.char_embedding_size)
    #pretrained_embedding = torch.from_numpy(np.load("char_embedding.npy")).float()
    #self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding) # [40, 32]
    
    ### A stack of three TF-like GRUs
    self.encoder_GRU = MultiGRU(self.hparams.char_embedding_size, self.hparams.cell_size)
    #self.encoder_GRU.from_pretrained(pretrained_encoder_dict)

    ### Concatentation of outputs of each GRU layer seperately
    self.encoder_dense_layer = nn.Linear(sum(self.hparams.cell_size), self.hparams.emb_size)
    #with torch.no_grad():
    #  self.encoder_dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-kernel.npy").T).float()) # [3584, 512]
    #  self.encoder_dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-bias.npy")).float()) # [512]
    
    ### Load pretrained weights
    #pretrained_decoder_dict = {}
    #for i in ["0","1","2"]:
    #  pretrained_decoder_dict["gru_{}".format(i)] = {}
    #  for j in ["candidate", "gates"]:
    #    for k in ["bias","kernel"]:
    #      file_name = "Decoder/decoder/multi_rnn_cell/cell_{}/gru_cell/{}/{}.npy".format(i,j,k).replace("/","-")
    #      pretrained_decoder_dict["gru_{}".format(i)]["{}-{}".format(j,k)] = np.load(file_name)

    
    ## Projection Layer that maps the final GRU output to token logits
    self.decoder_projection = nn.Linear( self.hparams.cell_size[-1] , self.hparams.voc_size, bias = False)
    #with torch.no_grad():
    #  self.decoder_projection.weight = nn.Parameter(torch.from_numpy(np.load("Decoder-decoder-dense-kernel.npy").T).float()) ## [2048, 40]

    ### A stack of three TF-like GRUs
    self.decoder_GRU = MultiGRU(self.hparams.char_embedding_size, self.hparams.cell_size)
    #self.decoder_GRU.from_pretrained(pretrained_decoder_dict)

    ### Layer that takes latent vector to concatenated GRU history (h0) inputs
    self.decoder_dense_layer = nn.Linear(self.hparams.emb_size, sum(self.hparams.cell_size))
    #with torch.no_grad():
    #  self.decoder_dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Decoder-dense-kernel.npy").T).float()) # [512, 3584 = 512 + 1024 + 2048]
    #  self.decoder_dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Decoder-dense-bias.npy")).float()) # [3584]
    
    self.optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = self.hparams.lr_decay_frequency)
    

  ### Embed some tokenized strings
  def encode(self, input_seqs, input_lens):
    """Encode sequences to latent vectors given lengths"""

    if( type(input_seqs) == torch.Tensor):
      batch_size = input_seqs.size()[0]
    else:
      batch_size = input_seqs.shape[0]
      input_seqs = torch.from_numpy(input_seqs).int()
    
    h0 = [torch.zeros(i) for i in self.hparams.cell_size]
    encoder_emb_inp = self.encoder_embedding(input_seqs)
    embeddings = torch.zeros([batch_size, self.hparams.emb_size])
    
    count = 0
    for compound_emb, length in zip(encoder_emb_inp, input_lens):
      encoder_state = h0
      for i in range(length):
        symbol = compound_emb[i]
        encoder_outputs, encoder_state = self.encoder_GRU(symbol, encoder_state)
      embeddings[count, :] = torch.tanh(self.encoder_dense_layer(torch.cat(encoder_state)))
      count += 1

    return embeddings

  def decode(self, input_vecs):
    """Convert latent vector to tokenized representation"""

    if( type(input_vecs) == torch.Tensor):
      batch_size = input_vecs.size()[0]
    else:
      batch_size = input_vecs.shape[0]
      input_vecs = torch.from_numpy(input_vecs).float()
    
    y = self.decoder_dense_layer(input_vecs)

    ## Size cum sum
    sizcs = [sum(self.hparams.cell_size[:i]) for i in range(len(self.hparams.cell_size)+1)]
    h0 = [ y[:,sizcs[i]:sizcs[i+1]] for i in range(len(sizcs)-1)]

    x = np.array([self.decode_vocabulary['<s>']] * batch_size)
    x = torch.from_numpy(x).int()
    x = self.encoder_embedding(x)
 
    all_strings = []
    batch_logit_tensor = torch.zeros([batch_size, self.hparams.voc_size , self.hparams.max_string_length - 1])  ### 10, 40, 149

    for i in range(batch_size): 
      output_string = ""
      last_argmax = self.decode_vocabulary['<s>']
      count = 0
      xx, hh = x[i], [ h0[j][i] for j in range(len(self.hparams.cell_size))]
      
      while last_argmax != 0 and count < self.hparams.max_string_length - 1:
        output, hh = self.decoder_GRU(xx, hh)
        logits = self.decoder_projection(output)
        batch_logit_tensor[i, :, count] = logits
        probs = torch.argmax(logits)
        last_argmax = int(probs.detach().numpy()) 
        token = self.decode_vocabulary_inv[last_argmax]
        output_string += token
        xx = self.encoder_embedding(probs)
        count += 1
      all_strings.append(output_string.replace("</s>",""))
    return batch_logit_tensor, all_strings

  def save(self, file_name = None):
    """ Save model state to file, or default in hparams"""
    if(file_name):
      torch.save(self.state_dict(), file_name)
    else:
      torch.save(self.state_dict(), self.hparams.sav_dir + "savfile.sav")

  def load(self, file_name = None):
    """Load model state from file, or default in hparams"""
    if(file_name):
      self.load_state_dict(torch.load(file_name))
    else:
      self.load_state_dict(torch.load(self.hparams.sav_dir + "savfile.sav"))

  def calc_loss(self, target_len, target_seq, logits):
    """Return the loss shifted sequence cross entropy loss over the mini-batch"""
    shifted_target_len = target_len - 1
    shifted_target_seq = target_seq[0:,1:].long()
    target_mask = torch.arange(target_seq.size()[1] - 1)[None, :] < shifted_target_len[:, None]
    target_mask = target_mask / torch.sum(target_mask)
    loss = F.cross_entropy(logits, shifted_target_seq, reduction = 'none')   ### grab from the file ## Check the order
    loss = torch.sum(loss * target_mask)
    return loss

  def train(self, dataset):
    """Train on the given dataset, according to the hparams"""

    for epoch in range(self.hparams.num_epochs):
      for step, (seq1, seq1_len) in tqdm( enumerate(dataset), total = len(dataset)):

        input_seq = torch.tensor(seq1)
        input_len = torch.tensor(seq1_len)
        target_seq = torch.tensor(seq1)
        target_len = torch.tensor(seq1_len)

        embedding = self.encode( input_seq, input_len )
        logits, strings = self.decode(embedding)
        loss = self.calc_loss(target_len, target_seq, logits)
        loss.backward()
        self.optimizer.step()

        ### Check perforemance and save a checkpoint of the model
        if( step % self.hparams.summary_freq and step != 0 ):
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
          torch.save(self.state_dict(), self.hparams.save_file)


if __name__ == "__main__":
  train()

