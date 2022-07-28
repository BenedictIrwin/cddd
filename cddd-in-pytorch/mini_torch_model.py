import torch
import torch.nn.functional as F

class MiniCDDDInference(torch.nn.Module):
  """NoisyGRUSeq2SeqWithFeatures"""
  def __init__(self):
    super().__init__()

    self.embedding_size = 512
    

    ### Each GRU Cell has 6 weights and 6 biases (or fewer if bias = False)
    self.encoder_GRU_Cell_1 = torch.nn.GRUCell(input_size, hidden_size, bias = True)
    self.encoder_GRU_Cell_2 = torch.nn.GRUCell(input_size, hidden_size, bias = True)
    self.encoder_GRU_Cell_3 = torch.nn.GRUCell(input_size, hidden_size, bias = True)
   
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

    ### Load up all of the TF weights
    # Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/bias [512]
    tf_gru_c_b = np.load()
    # Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/candidate/kernel [544, 512]
    # Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/gates/bias [1024]
    # Encoder/rnn/multi_rnn_cell/cell_0/gru_cell/gates/kernel [544, 1024]

    # Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/bias [1024]
    # Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/candidate/kernel [1536, 1024]
    # Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/gates/bias [2048]
    # Encoder/rnn/multi_rnn_cell/cell_1/gru_cell/gates/kernel [1536, 2048]

    # Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/candidate/bias [2048]
    # Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/candidate/kernel [3072, 2048]
    # Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/gates/bias [4096]
    # Encoder/rnn/multi_rnn_cell/cell_2/gru_cell/gates/kernel [3072, 4096]


    ## This creates some problems, because the weights are summed in TF, so we will either not use biases, or set
    ## degenerate weights to 0 in pytorch to accomodate.

    ~GRUCell.weight_ih (torch.Tensor) – the learnable input-hidden weights, of shape (3*hidden_size, input_size)
    ~GRUCell.weight_hh (torch.Tensor) – the learnable hidden-hidden weights, of shape (3*hidden_size, hidden_size)
    ~GRUCell.bias_ih – the learnable input-hidden bias, of shape (3*hidden_size)
    ~GRUCell.bias_hh – the learnable hidden-hidden bias, of shape (3*hidden_size)

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



    self.encoder_GRU = torch.nn.GRU(input_size = 544, hidden_size = 512, num_layers = 3) 

    self.dense_layer = nn.Linear( , self.embeding_size)
    self.dense_layer.weight = np.load("encoder_dense_weight.npy")
    self.dense_layer.bias = np.load("encoder_dense_bias.npy")

    self.features =


  ### Embed some tokenized strings
  def forward(self, input_seqs):
    """encode"""
    h0 = torch.zeros()
    encoder_outputs, encoder_state = self.sencoder_GRU(encoder_emb_inp, h0)
    emb = self.dense_layer(torch.cat(encoder_state, dim = 1))
    emb = F.tanh(emb)

### Some examples
test_inputs = np.load("test_in_seq.npy")
test_input_len = np.load("test_in_len.npy")
test_outputs = np.load("test_output_embeddings.npy")


### Call the model
model = MiniCDDDInference()

outputs = model(test_inputs)

