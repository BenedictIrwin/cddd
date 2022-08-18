from cddd.inference import InferenceModel
import os


from models.py import *


def test_model_from_pretrained():
  """ Simple test of using model"""

  model_dir = ""
  ## Check of os.enviroment_var["CUDA DEVICES"] ---> True
  for use_gpu in [False, True]:
    infer_model = InferenceModel(model_dir = model_dir, use_gpu = use_gpu, batch_size = 3, cpu_threads = 1)
    smiles = ["CCCCC","C","CC","c1ccccc1"]
    result = infer_model.seq_to_emb(smiles)
    print(result)

    assert 1 == 1 ### To build result based check

def test_model_init():
  """ Test the initialisation of a model """

    hparams = {}
    print("TO REMOVE iterator and mode from model")
    model = NoisyGRUSeq2SeqWithFeatures("ENCODE", None, hparams)

    ### Do some checks
    

def test_TF1_embedding_equivalence():
  """ Test the embedded vectors are close to TF1 fixed values """

  answers_embeddings_TF1 = torch.from_numpy(np.load("test_output_embeddings.npy"))
 
  ### Get hparams

  ### Initalise model with pretrained weights

  ### Define dataset
  input_smiles = []

  ### Call model on smiles to get embeddings

  assert torch.allclose()

  ### Reconstruct the SMILES

  assert output_smiles == input_smiles



def test_training():

  ### Get hparams

  ### Initialise model

  ### Define dataset

  ### model.train


  pass
### Some examples
#test_inputs = np.load("test_in_seq.npy")
#test_input_len = np.load("test_in_len.npy")
#test_outputs = np.load("test_output_embeddings.npy")

### Call the model
#encoder = MiniCDDDInference()
#outputs = encoder(test_inputs, test_input_len)

#print(outputs)
#print(outputs.shape)
#print(test_outputs)


#decoder = MiniCDDDDecoder()
#outputs = decoder(outputs)  ## Input N x 512 vectors... generate compounds




### Run tests
test_model_from_pretrained()
test_training()



