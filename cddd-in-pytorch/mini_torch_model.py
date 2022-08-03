import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiGRU(torch.nn.Module):
  """Stacked version of MultiGRU with TF like weight structure"""
  def __init__(self, voc_size, latent_vec_sizes):
    super(MultiGRU, self).__init__()
    #self.embedding = nn.Embedding(voc_size, 128)
    self.gru_0 = nn.GRUCell(voc_size, latent_vec_sizes[0])
    self.gru_1 = nn.GRUCell(latent_vec_sizes[0], latent_vec_sizes[1])
    self.gru_2 = nn.GRUCell(latent_vec_sizes[1], latent_vec_sizes[2])
    #self.linear = nn.Linear(latent_vec_sizes[2], voc_size)
  
  def forward(self, x, h):
    #x = self.embedding(x)
    h_out = Variable(torch.zeros(h.size()))
    x = h_out[0] = self.gru_0(x, h[0])
    x = h_out[1] = self.gru_1(x, h[1])
    x = h_out[2] = self.gru_2(x, h[2])
    #x = self.linear(x)
    return x, h_out

  def from_pretrained(self, pretrained):
    """Loads up from gate and candidate kernels and biases K_g, K_c, b_g, b_c"""
    self.gru_0.from_pretrained( pretrained["gru_0"] )
    self.gru_1.from_pretrained( pretrained["gru_1"] )
    self.gru_2.from_pretrained( pretrained["gru_2"] )

  #def init_h(self, batch_size, latent_vectors):
  #  # Initial cell state is zero
  #  #x = Variable(torch.zeros(3, batch_size, 330))
  #  if len(latent_vectors) < 330:
  #    scale_net = math.ceil(330/len(latent_vectors))
  #    latent_vectors = latent_vectors.repeat(1, scale_net)
  #  return Variable(latent_vectors.repeat(3, 1, 1))

class GRUTensorflowLike(torch.nn.Module):
  # Imitates the GRU cell of the tensorflow implementation, with zero dropout
  def __init__(self):
    super().__init__()
    #self.input_size = input_size
    #self.weight_ih = weight_ih.T
    #self.weight_hh = weight_hh.T

    #print("Possibly adjust this so you can just plop in the TF K_g and K_c etc. and it automatically assigns all of the weights.")
    #print("SORT OUT BIASES in GRUTensorflowLike!")
    #exit()
    self._activation = torch.tanh

  def from_pretrained(self, dictionary):
    """ Split up the K_g, K_c, b_g, b_c weights correctly """
    ## Get sizes
    #siz_h = dictionary["gates-kernel"].shape[1]/2
    #siz_i = dictionary["gates-kernel"].shape[0] - siz_h

    ## Extract kernels
    #self.W_ir = dictionary["gates-kernel"][:siz_i, :siz_h].T
    #self.W_iz = dictionary["gates-kernel"][siz_i:, :siz_h].T
    #self.W_hr = dictionary["gates-kernel"][:siz_i, siz_h:].T
    #self.W_hz = dictionary["gates-kernel"][siz_i:, siz_h:].T
    #self.W_in = dictionary["candidate-kernel"][:siz_i,:].T
    #self.W_hn = dictionary["candidate-kernel"][siz_i:,:].T

    ## Extract Biases (forcing degenerate terms to zero)
    #self.b_ir = 
    #self.b_hr =
    #self.b_iz = 
    #self.b_hz = 
    #self.b_in = 
    #self.b_hn = 
   
    ## TF like variables
    self._gate_kernel = dictionary["gates-kernel"]
    self._gate_bias = dictionary["gates-bias"]
    self._candidate_kernel = dictionary["candidate-kernel"]
    self._candidate_bias = dictionary["candidate-bias"]


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
    
    ### TF call #######################################################
    gate_inputs = torch.matmul(torch.cat([x, h], dim = 1), self._gate_kernel)
    gate_inputs = gate_inputs + self._gate_bias

    value = torch.sigmoid(gate_inputs)
    r, z = torch.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * h

    candidate = torch.matmul(torch.cat([inputs, r_state], dim = 1), self._candidate_kernel)
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

    print(pretrained_gru_dict)
    exit()

    ### Load up all of the TF weights
    #string = "Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias" # [512]
    #tf_gru_0_c_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel" # [544, 512]
    #tf_gru_0_c_k = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias" # [1024]
    #tf_gru_0_g_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel" # [544, 1024]
    #tf_gru_0_g_k = np.load(string.replace("/","-")+".npy")

    #string = "Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias" # [1024]
    #tf_gru_1_c_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel" # [1536, 1024]
    #tf_gru_1_c_k = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias" # [2048]
    #tf_gru_1_g_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel" # [1536, 2048]
    #tf_gru_1_g_k = np.load(string.replace("/","-")+".npy")

    #string = "Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/candidate/bias" # [2048]
    #tf_gru_2_c_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernel" # [3072, 2048]
    #tf_gru_2_c_k = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/gates/bias" # [4096]
    #tf_gru_2_g_b = np.load(string.replace("/","-")+".npy")
    #string = "Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernel" # [3072, 4096]
    #tf_gru_2_g_k = np.load(string.replace("/","-")+".npy")


    ## This creates some problems, because the weights are summed in TF, so we will either not use biases, or set
    ## degenerate weights to 0 in pytorch to accomodate.
        
        ## gate_kernel of shape [input_depth + self._num_units, 2 * self.num_units] i.e. 4 of the matrices
        ## candidate_kernel of shape [input_depth + self.num_units, self.num_units]
        ## gate_bias = 2* self._num_units 
        ## candidate_bias = self._num_units 
    ### Assume TF block matrix K_g ~~~~~~~~~
    #          [W_ir.T|W_iz.T]
    #  [x]|[h].--------------- + b_g //// b_g = sum of biases?
    #          [W_hr.T|W_hz.T]
    ###########################################
    #####################################
    ### Assume TF block matrix K_c ~~~~~~~~
    #
    #          []
    #  [x|r*h].------   + b_c //// b_c = sum of biases?
    #          []
    #
    # NOT SURE WHETHER IT IS THE MATRIX OR THE TRANSPOSE?

    #~GRUCell.weight_ih (torch.Tensor) – the learnable input-hidden weights, of shape (3*hidden_size, input_size)
    #~GRUCell.weight_hh (torch.Tensor) – the learnable hidden-hidden weights, of shape (3*hidden_size, hidden_size)
    #~GRUCell.bias_ih – the learnable input-hidden bias, of shape (3*hidden_size)
    #~GRUCell.bias_hh – the learnable hidden-hidden bias, of shape (3*hidden_size)
    # Attributes:
    # weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
    # (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
    # Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
    # weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
    # (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
    # bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
    # (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
    # bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
    # (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`
  
    """
    ### Does it stack like r,z,n -- > Yes
    print(W_ir_0.shape)
    print(W_iz_0.shape)
    print(W_in_0.shape)

    self.encoder_GRU_Cell_0.weight_ih = np.hstack([W_ir_0, W_iz_0, W_in_0])
    self.encoder_GRU_Cell_0.weight_hh = np.stack([W_hr_0, W_hz_0, W_hn_0], axis = 0)
    self.encoder_GRU_Cell_0.bias_ih   = np.stack([b_ir_0, b_iz_0, b_in_0], axis = 0)
    self.encoder_GRU_Cell_0.bias_hh   = np.stack([b_hr_0, b_hz_0, b_hn_0], axis = 0)
    
    self.encoder_GRU_Cell_1.weight_ih = np.stack([W_ir_1, W_iz_1, W_in_1], axis = 0)
    self.encoder_GRU_Cell_1.weight_hh = np.stack([W_hr_1, W_hz_1, W_hn_1], axis = 0)
    self.encoder_GRU_Cell_1.bias_ih   = np.stack([b_ir_1, b_iz_1, b_in_1], axis = 0)
    self.encoder_GRU_Cell_1.bias_hh   = np.stack([b_hr_1, b_hz_1, b_hn_1], axis = 0)
    
    self.encoder_GRU_Cell_2.weight_ih = np.stack([W_ir_2, W_iz_2, W_in_2], axis = 0)
    self.encoder_GRU_Cell_2.weight_hh = np.stack([W_hr_2, W_hz_2, W_hn_2], axis = 0)
    self.encoder_GRU_Cell_2.bias_ih   = np.stack([b_ir_2, b_iz_2, b_in_2], axis = 0)
    self.encoder_GRU_Cell_2.bias_hh   = np.stack([b_hr_2, b_hz_2, b_hn_2], axis = 0)
    

    ### Requires GRU Cells due to unequal layer sizes
    ### Each GRU Cell has 6 weights and 6 biases (or fewer if bias = False)
    self.encoder_GRU_Cell_0 = torch.nn.GRUTensorflowLike(32, )#512, bias = True)
    self.encoder_GRU_Cell_1 = torch.nn.GRUTensorflowLike(512, )#1024, bias = True)
    self.encoder_GRU_Cell_2 = torch.nn.GRUTensorflowLike(1024, )#2048, bias = True)
    """

    ### TF call #######################################################
    #gate_inputs = math_ops.matmul(
    #array_ops.concat([inputs, state], 1), self._gate_kernel)
    #gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    #value = math_ops.sigmoid(gate_inputs)
    #r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    #r_state = r * state

    #candidate = math_ops.matmul(
    #    array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    #candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    #c = self._activation(candidate)
    #new_h = u * state + (1 - u) * c
    #return new_h, new_h
    ###################################################################
    # TF VERSION OF THINGS  u = z, state = h, input = x, c = n
    # val = [x|h] . K_g
    # val = val + b_g
    # v = sigma(val)
    # r, z = split( v, ... )
    # r_state = r * state
    # c = [inputs|r_state] . K_c
    # c = c + b_c
    # c = activation/tanh(c)
    # new_h = z * state + (1 - z) * c


    ###################################################################
    # PYTORCH VERSION OF THINGS (maths)
    # r = sigma( W_ir x + b_ir + W_hr h + b_hr )
    # z = sigma( W_iz x + b_iz + W_hz + b_hz )
    # n = tanh( W_in x + b_in + r *( W_hn h + b_hn ) )
    # h' = (1-z) * n + z * h
    #######################################
    # PYTORCH CODE VERSION OF THINGS
    # 
    #
    #
    #
    #
    #
    ############################################################



    #self.encoder_GRU = torch.nn.GRU(input_size = 32, hidden_size = 512, num_layers = 3) 

    
    #self.encoder_embedding = nn.Embedding(40, 32)
    pretrained_embedding = torch.from_numpy(np.load("char_embedding.npy")).float()
    self.encoder_embedding = nn.Embedding.from_pretrained(pretrained_embedding) # [40, 32]
    
    ### A stack of three TF-like GRUs
    self.encoder_GRU = MultiGRU(32,[512,1024,2048])
    self.encoder_GRU.from_pretrained(...)

    ### Concatentation of outputs of each GRU layer seperately
    self.dense_layer = nn.Linear(512 + 1024 + 2048, self.latent_size)
    with torch.no_grad():
      self.dense_layer.weight = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-kernel.npy")).float()) # [3584, 512]
      self.dense_layer.bias = nn.Parameter(torch.from_numpy(np.load("Encoder-dense-bias.npy")).float()) # [512]
    
    #self.features =


  ### Embed some tokenized strings
  def forward(self, input_seqs):
    """encode"""
    input_seqs = torch.from_numpy(input_seqs).int()
    h0 = torch.zeros(self.latent_size) ## Or should it be random?
    encoder_emb_inp = self.encoder_embedding(input_seqs)
    print(encoder_emb_inp)
    print(encoder_emb_inp.size())
    encoder_outputs, encoder_state = self.encoder_GRU(encoder_emb_inp, h0)
    print(encoder_outputs)
    print(encoder_outputs.size())
    exit()
    emb = self.dense_layer(torch.cat(encoder_state, dim = 1))
    emb = F.tanh(emb)

### Some examples
test_inputs = np.load("test_in_seq.npy")
test_input_len = np.load("test_in_len.npy")
test_outputs = np.load("test_output_embeddings.npy")

print(test_inputs)
print(test_inputs.shape)

### Call the model
model = MiniCDDDInference()

outputs = model(test_inputs)

