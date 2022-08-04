import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import Variable

class MultiGRU(torch.nn.Module):
  """Stacked version of MultiGRU with TF like weight structure"""
  def __init__(self, voc_size, latent_vec_sizes):
    super(MultiGRU, self).__init__()
    #self.embedding = nn.Embedding(voc_size, 128)
    self.gru_0 = TFGRUCell(voc_size, latent_vec_sizes[0])
    self.gru_1 = TFGRUCell(latent_vec_sizes[0], latent_vec_sizes[1])
    self.gru_2 = TFGRUCell(latent_vec_sizes[1], latent_vec_sizes[2])
    #self.linear = nn.Linear(latent_vec_sizes[2], voc_size)
  
  def forward(self, x, h):
    #print(x)
    #print(h)
    #print(x.size())
    #print([i.size() for i in h])
    #print(torch.cat([x,h[0]]))
    #x = self.embedding(x)
    #h_out = Variable(torch.zeros(h.size()))
    h_out = [0,0,0]
    x = h_out[0] = self.gru_0(x, h[0])
    #!!!return x, h_out
    x = h_out[1] = self.gru_1(x, h[1])
    x = h_out[2] = self.gru_2(x, h[2])
    #x = self.linear(x)
    return x, h_out

  def from_pretrained(self, pretrained):
    """Loads up from gate and candidate kernels and biases K_g, K_c, b_g, b_c"""
    self.gru_0.from_pretrained( pretrained["gru_0"] )
    self.gru_1.from_pretrained( pretrained["gru_1"] )
    self.gru_2.from_pretrained( pretrained["gru_2"] )

class TFGRUCell(torch.nn.Module):
  # Imitates the GRU cell of the tensorflow implementation, with zero dropout
  def __init__(self, in_size, out_size):
    super().__init__()
    self._activation = torch.tanh
    self.in_size = in_size
    self.out_size = out_size
    #### TO DO, ADD initialised torch.tensors
    #### Considering initialization

  def from_pretrained(self, dictionary):
    """ Split up the K_g, K_c, b_g, b_c weights correctly """
    with torch.no_grad():
      self._gate_kernel      = nn.Parameter(torch.from_numpy(dictionary["gates-kernel"]).float())
      self._gate_bias        = nn.Parameter(torch.from_numpy(dictionary["gates-bias"]).float())
      self._candidate_kernel = nn.Parameter(torch.from_numpy(dictionary["candidate-kernel"]).float())
      self._candidate_bias   = nn.Parameter(torch.from_numpy(dictionary["candidate-bias"]).float())

  def forward(self, x, h):
    #n = self.input_size
    ##x_r = torch.matmul(self.W_ir, x) + torch.matmul(self.W_hr, h)
    ##r = torch.sigmoid( x_r + self.b_ir + self.b_hr)
    ##x_z = torch.matmul(self.W_iz, x) + torch.matmul(self.W_hz, h)
    ##z = torch.sigmoid( x_z + self.b_iz + self.b_hz)
    ##x_n = torch.matmul(self.W_in, x) + torch.matmul(self.W_hn, r * h)
    ##n = torch.tanh( x_n + self.b_in + r * self.b_hn)
    #x_z = torch.matmul(inputs, self.weight_ih[:, :n])
    #x_r = torch.matmul(inputs, self.weight_ih[:, n : n * 2])
    #x_h = torch.matmul(inputs, self.weight_ih[:, n * 2 :])
    #recurrent_z = torch.matmul(h_tm1, self.weight_hh[:, :n])
    #recurrent_r = torch.matmul(h_tm1, self.weight_hh[:, n : n * 2])
    #z = torch.sigmoid(x_z + recurrent_z)
    #r = torch.sigmoid(x_r + recurrent_r)
    #recurrent_h = torch.matmul(r * h_tm1, self.weight_hh[:, n * 2 :])
    #hh = torch.tanh(x_h + recurrent_h)
    ##h = z * h + (1 - z) * n
   
    #print(x.size())
    #print(h.size())
    #input("INPUT SIZES ABOVE")
    ### TF call #######################################################
    #gate_inputs = torch.matmul(torch.cat([x, h], dim = 1), self._gate_kernel)
    gate_inputs = torch.matmul(torch.cat([x, h]), self._gate_kernel)
    gate_inputs = gate_inputs + self._gate_bias

    value = torch.sigmoid(gate_inputs)
    #print(value)
    #print(value.size())
    #print("SIZES",self.in_size, self.out_size)
    #input("VALUE ABOVE")
    r, z = value[:self.out_size], value[self.out_size:]

    r_state = r * h
    #print(x.size())
    #print(r_state.size())
    #input("VERIFY ABOVE")

    candidate = torch.matmul(torch.cat([x, r_state]), self._candidate_kernel)
    candidate = candidate + self._candidate_bias

    n = self._activation(candidate)
    h = z * h + (1 - z) * n
    return h

class MiniCDDDInference(torch.nn.Module):
  """NoisyGRUSeq2SeqWithFeatures"""
  def __init__(self):
    super().__init__()

    self.embedding_size = 32
    self.latent_size = 512
    
   
    #self.encoder_GRU = nn.Sequential(...,...,...) ?

    #### Presumably the size 3*hidden, input -> W_ir, W_iz, W_in stacked, etc.
    
    #### https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/ops/rnn_cell_impl.py#L281-L309
    #### In TF 1.15 GRUCell has two matrices
        ## gate_kernel of shape [input_depth + self._num_units, 2 * self.num_units] i.e. 4 of the matrices
        ## candidate_kernel of shape [input_depth + self.num_units, self.num_units]
        ## gate_bias = 2* self._num_units 
        ## candidate_bias = self._num_units 
        ## i.e. 3*distinct biases... 
   
    ## Encoder cell 0 in TF candidate bias = 512, gates bias = 1024 (i.e. num_units = 512)
    ## Encoder cell 1 in TF candidate bias = 1024, gates bias = 2048 (i.e. num_units = 1024)
    ## Encoder cell 2 in TF candidate bias = 2048, gates bias = 4096 (i.e. num_units = 2048)
    
    ## Encoder cell 0 in TF candidate kernel = [544, 512] === [32 + 512, 512]
    ## Encoder cell 1 in TF candidate kernel = [1536, 1024] === [512 + 1024, 1024]
    ## Encoder cell 2 in TF candidate kernel = [3072, 2048] === [1024 + 2048, 2048]

    pretrained_gru_dict = {}
    for i in ["0","1","2"]:
      pretrained_gru_dict["gru_{}".format(i)] = {}
      for j in ["candidate", "gates"]:
        for k in ["bias","kernel"]:
          file_name = "Encoder/rnn/multi_rnn_cell/cell_{}/gru_cell/{}/{}.npy".format(i,j,k).replace("/","-")
          pretrained_gru_dict["gru_{}".format(i)]["{}-{}".format(j,k)] = np.load(file_name)

    
    #self.encoder_embedding = nn.Embedding(40, 32)
    pretrained_embedding = torch.from_numpy(np.load("char_embedding.npy")).float()
    self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding) # [40, 32]
    
    ### A stack of three TF-like GRUs
    self.encoder_GRU = MultiGRU(32,[512,1024,2048])
    self.encoder_GRU.from_pretrained(pretrained_gru_dict)

    ### Concatentation of outputs of each GRU layer seperately
    self.dense_layer = nn.Linear(512 + 1024 + 2048, self.latent_size)
    with torch.no_grad():
      self.dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-kernel.npy").T).float()) # [3584, 512]
      self.dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-bias.npy")).float()) # [512]
    
    #self.features =


  ### Embed some tokenized strings
  def forward(self, input_seqs, input_lens):
    """encode"""
    input_seqs = torch.from_numpy(input_seqs).int()
    #h0 = torch.zeros([3,self.latent_size]) ## Or should it be random?
    h0 = [torch.zeros(512), torch.zeros(1024), torch.zeros(2048)] ## Or should it be random?
    #print(h0)
    encoder_emb_inp = self.encoder_embedding(input_seqs)
    #print(encoder_emb_inp)
    ##print(encoder_emb_inp.size())
    #exit()


    outputs = []
    states = []
    for compound_emb, length in zip(encoder_emb_inp, input_lens):
      print("###",length,compound_emb.size())
      #symbol_emb = compound_emb[0] ## Get the first token
      encoder_state = h0
      #for i in range(length):
      #for i in range(len(compound_emb)):
      for i in range(length):
        symbol = compound_emb[i]
        encoder_outputs, encoder_state = self.encoder_GRU(symbol, encoder_state)
        #print(i,encoder_outputs)
      #print(encoder_state.size())
      #  print(encoder_state)
      #  for i in encoder_state:
      #    print(i.size())
      #    break
      #  for i in encoder_state:
      #    print(i)
      #    break
      #  exit()
      #   print(encoder_state)
      emb = self.dense_layer(torch.cat(encoder_state))
      emb = torch.tanh(emb)
      print(emb.size())
      print(emb)
      input()

### Some examples
test_inputs = np.load("test_in_seq.npy")
test_input_len = np.load("test_in_len.npy")
test_outputs = np.load("test_output_embeddings.npy")

print(test_inputs)
print(test_inputs.shape)

### Call the model
model = MiniCDDDInference()

outputs = model(test_inputs, test_input_len)

