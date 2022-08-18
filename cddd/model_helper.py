"""Helper functions that build the translation model with a corroponding graph and session."""
from collections import namedtuple
import tensorflow as tf
from cddd import models
from cddd import input_pipeline

def build_models(hparams, modes=["TRAIN", "EVAL", "ENCODE"]):
    """Helper function to build a translation model for one or many different modes.

    Args:
        hparams: Hyperparameters defined in file or flags.
        modes: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
        Can be a list if multiple models should be build.
    Returns:
        One model or a list of multiple models.
    """
    model = getattr(models, hparams.model)
    input_pipe = getattr(input_pipeline, hparams.input_pipeline)
    model_list = []
    if isinstance(modes, list):
        for mode in modes:
            model_list.append(create_model(mode, model, input_pipe, hparams))
        return tuple(model_list)
    else:
        model = create_model(modes, model, input_pipe, hparams)
        return model

Model = namedtuple("Model", ("graph", "model", "sess"))

def create_model(mode, model_creator, input_pipeline_creator, hparams):
    if mode in ["TRAIN", "EVAL"]:
      #input_pipe = input_pipeline_creator(mode, hparams)
      #input_pipe.make_dataset_and_iterator()
      #iterator = input_pipe.iterator
      iterator = None ### to be removed
      #print("CREATE MODEL: TRAIN EVAL NOT SUPPORTED!")
      #exit()
    else:
      iterator = None
    model = model_creator(mode = mode, iterator = iterator, hparams = hparams)
    return Model(graph = None, model = model, sess = None)

