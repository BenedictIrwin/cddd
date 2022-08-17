from cddd.inference import InferenceModel
import os

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

def test_training():

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



