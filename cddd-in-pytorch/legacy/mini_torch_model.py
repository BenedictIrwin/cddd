import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import Variable

class MultiGRU(torch.nn.Module):
  """Stacked version of MultiGRU with TF like weight structure"""
  def __init__(self, voc_size, latent_vec_sizes):
    super(MultiGRU, self).__init__()
    self.gru_0 = TFGRUCell(voc_size, latent_vec_sizes[0])
    self.gru_1 = TFGRUCell(latent_vec_sizes[0], latent_vec_sizes[1])
    self.gru_2 = TFGRUCell(latent_vec_sizes[1], latent_vec_sizes[2])
  
  def forward(self, x, h):
    """Run the three GRU cells with zero input"""
    h_new = [0,0,0]
    x = h_new[0] = self.gru_0(x, h[0])
    x = h_new[1] = self.gru_1(x, h[1])
    x = h_new[2] = self.gru_2(x, h[2])
    return x, h_new

  def from_pretrained(self, pretrained):
    """Loads up from gate and candidate kernels and biases K_g, K_c, b_g, b_c"""
    self.gru_0.from_pretrained(pretrained["gru_0"])
    self.gru_1.from_pretrained(pretrained["gru_1"])
    self.gru_2.from_pretrained(pretrained["gru_2"])

class TFGRUCell(torch.nn.Module):
  """A GRU Cell made to work exactly as the TF version"""
  def __init__(self, in_size, out_size):
    super().__init__()
    self._activation = torch.tanh
    self.in_size = in_size
    self.out_size = out_size

  def from_pretrained(self, dictionary):
    """ Split up the K_g, K_c, b_g, b_c weights correctly """
    with torch.no_grad():
      self._gate_kernel      = nn.Parameter(torch.from_numpy(dictionary["gates-kernel"]).float())
      self._gate_bias        = nn.Parameter(torch.from_numpy(dictionary["gates-bias"]).float())
      self._candidate_kernel = nn.Parameter(torch.from_numpy(dictionary["candidate-kernel"]).float())
      self._candidate_bias   = nn.Parameter(torch.from_numpy(dictionary["candidate-bias"]).float())

  def forward(self, x, h):
    """ Calculate GRU logic using block matrices """
    gate_inputs = torch.matmul(torch.cat([x, h]), self._gate_kernel) + self._gate_bias
    value = torch.sigmoid(gate_inputs)
    r, z = value[:self.out_size], value[self.out_size:]
    r_state = r * h
    candidate = torch.matmul(torch.cat([x, r_state]), self._candidate_kernel) + self._candidate_bias
    return z * h + (1 - z) * self._activation(candidate)

class MiniCDDDInference(torch.nn.Module):
  """NoisyGRUSeq2SeqWithFeatures"""
  def __init__(self):
    super().__init__()

    self.voc_emb_size = 32
    self.gru_layer_sizes = [512, 1024, 2048]
    self.latent_size = 512

    ### Load pretrained weights
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
    self.encoder_GRU = MultiGRU(self.voc_emb_size, self.gru_layer_sizes)
    self.encoder_GRU.from_pretrained(pretrained_gru_dict)

    ### Concatentation of outputs of each GRU layer seperately
    self.dense_layer = nn.Linear(sum(self.gru_layer_sizes), self.latent_size)
    with torch.no_grad():
      self.dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-kernel.npy").T).float()) # [3584, 512]
      self.dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-bias.npy")).float()) # [512]

  ### Embed some tokenized strings
  def forward(self, input_seqs, input_lens):
    """encode"""
    
    input_seqs = torch.from_numpy(input_seqs).int()
    h0 = [torch.zeros(i) for i in self.gru_layer_sizes]
    encoder_emb_inp = self.encoder_embedding(input_seqs)

    embeddings = []
    for compound_emb, length in zip(encoder_emb_inp, input_lens):
      encoder_state = h0
      for i in range(length):
        symbol = compound_emb[i]
        encoder_outputs, encoder_state = self.encoder_GRU(symbol, encoder_state)
      emb = self.dense_layer(torch.cat(encoder_state))
      embeddings.append(torch.tanh(emb).detach().numpy())

    return np.array(embeddings)

class MiniCDDDDecoder(torch.nn.Module):
  """NoisyGRUSeq2SeqWithFeatures"""
  def __init__(self):
    super().__init__()

    self.voc_emb_size = 32
    self.voc_size = 40
    self.gru_layer_sizes = [512, 1024, 2048]
    self.latent_size = 512
    
    ### Load pretrained weights
    pretrained_gru_dict = {}
    for i in ["0","1","2"]:
      pretrained_gru_dict["gru_{}".format(i)] = {}
      for j in ["candidate", "gates"]:
        for k in ["bias","kernel"]:
          file_name = "Decoder/decoder/multi_rnn_cell/cell_{}/gru_cell/{}/{}.npy".format(i,j,k).replace("/","-")
          pretrained_gru_dict["gru_{}".format(i)]["{}-{}".format(j,k)] = np.load(file_name)

    
    ## Projection Layer that maps the final GRU output to token logits
    self.decoder_dense = nn.Linear( self.gru_layer_sizes[-1] , self.voc_size, bias = False)
    with torch.no_grad():
      self.decoder_dense.weight = nn.Parameter(torch.from_numpy(np.load("Decoder-decoder-dense-kernel.npy").T).float()) ## [2048, 40]

    ## decoder_embedding : uses the same weights as the encoder embedding
    pretrained_embedding = torch.from_numpy(np.load("char_embedding.npy")).float()
    self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding) # [40, 32]

    ### A stack of three TF-like GRUs
    self.decoder_GRU = MultiGRU(self.voc_emb_size, self.gru_layer_sizes)
    self.decoder_GRU.from_pretrained(pretrained_gru_dict)

    ### Layer that takes latent vector to concatenated GRU history (h0) inputs
    self.dense_layer = nn.Linear(self.latent_size, sum(self.gru_layer_sizes))
    with torch.no_grad():
      self.dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Decoder-dense-kernel.npy").T).float()) # [512, 3584 = 512 + 1024 + 2048]
      self.dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Decoder-dense-bias.npy")).float()) # [3584]

    vocab_file = "cddd/data/indices_char.npy"
    self.decode_vocabulary = { v: k for k, v in np.load(vocab_file, allow_pickle=True).item().items() }
    self.decode_vocabulary_inv = { k: v for k, v in np.load(vocab_file, allow_pickle=True).item().items() }

  def forward(self, input_vecs):

    y = torch.from_numpy(input_vecs).float()
    y = self.dense_layer(y)
    h0 = [y[:,:512], y[:,512:512+1024], y[:,512+1024:]]

    ### Or do [39] * batch_size?
    x = np.array([39,39,39,39,39,39])
    x = torch.from_numpy(x).int()
    x = self.encoder_embedding(x) ## x is length 32 using character encoding
   
    for i in range(len(x)):
      output_string = "<s>"
      last_argmax = 39 ## I.e. start char
      count = 0
      xx, hh = x[i], [ h0[0][i], h0[1][i], h0[2][i]]
      while last_argmax != 0 and count < 50:
        output, hh = self.decoder_GRU(xx, hh)
        logits = self.decoder_dense(output)  ### Taking from 2048 to [logits in 40]
        probs = torch.argmax(logits)  ### one hot in 40
        last_argmax = int(probs.detach().numpy()) 
        token = self.decode_vocabulary_inv[last_argmax]
        output_string += token
        xx = self.encoder_embedding(probs)
        count += 1
      print(output_string)
    exit()


### Some examples
test_inputs = np.load("test_in_seq.npy")
test_input_len = np.load("test_in_len.npy")
test_outputs = np.load("test_output_embeddings.npy")

### Call the model
encoder = MiniCDDDInference()
outputs = encoder(test_inputs, test_input_len)

print(outputs)
print(outputs.shape)
#print(test_outputs)


decoder = MiniCDDDDecoder()
outputs = decoder(outputs)  ## Input N x 512 vectors... generate compounds





