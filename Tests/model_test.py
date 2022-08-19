from cddd.inference import InferenceModel
import os


from cddd.models import *
from cddd.hyperparameters import create_hparams
from cddd.data_structs import DatasetWithFeatures  
from tqdm import tqdm

#def test_model_from_pretrained():
#  """ Simple test of using model"""
#
#  model_dir = ""
#  ## Check of os.enviroment_var["CUDA DEVICES"] ---> True
#  for use_gpu in [False, True]:
#    infer_model = InferenceModel(model_dir = model_dir, use_gpu = use_gpu, batch_size = 3, cpu_threads = 1)
#    smiles = ["CCCCC","C","CC","c1ccccc1"]
#    result = infer_model.seq_to_emb(smiles)
#    print(result)
#
#    assert 1 == 1 ### To build result based check

#def test_model_init():
#  """ Test the initialisation of a model """
#
#  #hparams = {}
#  print("TO REMOVE iterator and mode from model")
#  model = NoisyGRUSeq2SeqWithFeatures(hparams)

#  ### Do some checks
    

def test_TF1_embedding_equivalence():
  """ Test the embedded vectors are close to TF1 fixed values """

  answers_embeddings_TF1 = torch.from_numpy(np.load("test_output_embeddings.npy"))

  hparams = create_hparams()
  hparams.cell_size = [512, 1024, 2048]
  hparams.max_string_length = 150  ##### TO BE EDITED in hparams
  hparams.voc_size = 40
  hparams.emb_size = 512
  hparams.batch_size = 64 ## As longa as > 7 it will work

  model = NoisyGRUSeq2SeqWithFeatures(hparams)
  model.load("pretrained_cddd_model.sav")

  input_smiles = [
    "C[n+]1c2ccccc2c(N)c2ccccc21",
    "c1ccc2cnccc2c1",
    "Cc1c(N)cccc1[N+](=O)[O-]",
    "CCCNc1c([N+](=O)[O-])ccc(Cl)c1C(=O)O",
    "CNC(=O)N(C)N=O",
    "Nc1cccc2c1C(=O)c1ccccc1C2=O",
    "CC(=O)C1(C)OC12C=Cc1ccccc1C2=O"]
  
  data = DatasetWithFeatures(model, input_smiles)
  smiles, lens = data[0]
  output_embeddings = model.encode(smiles,lens)

  assert torch.allclose(output_embeddings, answers_embeddings_TF1, rtol = 1e-04, atol = 1e-06)

  _, output_smiles = model.decode(output_embeddings) 

  assert output_smiles == input_smiles


def test_training():
  """ A test to train a model from initialised state """

  ### Get hparams
  hparams = create_hparams()
  hparams.weights_file = None ### or hparams.load_file
  hparams.training_epochs = 1
  hparams.training_steps = 10
  hparams.save_file = "testing_training_sav" ## or /dev/null

  ### Initialise model
  model = NoisyGRUSeq2SeqWithFeatures(hparams)
  ### model - getattribute(hparams.model_type)(hparams) 

  ### Define dataset
  training_dataset = dataset_construction(hparams.training)

  ### model.train  This is likely to be pytorch.train
  model.train(training_dataset)

  assert 1 == 1 ### Coming up with a mteric i.e. training_loss decreases over epochs

def test_qsar():
  """ A test to run QSAR models using the active latent space """
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

test_TF1_embedding_equivalence()
#exit()

### Run tests
#test_model_from_pretrained()
test_training()



