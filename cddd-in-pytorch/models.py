"""Base translation model with different variations"""
import os
import shutil
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(ABC):
    """
    This is the base class for the translation model. Child class defines encode and decode
    architecture.

    Attribures:
        mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
        iterator: The iterator of the input pipeline.
        embedding_size: The size of the bottleneck layer which is later used as molecular
        descriptor.
        encode_vocabulary: Dictonary that maps integers to unique tokens of the
        input strings.
        decode_vocabulary: Dictonary that maps integers to unique tokens of the
        output strings.
        encode_voc_size: Number of tokens in encode_vocabulary.
        decode_voc_size: Number of tokens in decode_vocabulary.
        char_embedding_size: Number of Dimensiones used to encode the one-hot encoded tokens
        in a contineous space.
        global_step: Counter for steps during training.
        save_dir: Path to directory used to save the model and logs.
        checkpoint_path: Path to the model checkpoint file.
        batch_size: Number of samples per training batch.
        rand_input_swap: Flag to define if (for SMILES input) the input SMILES should be swapped
        randomly between canonical SMILES (usually output sequnce) and random shuffled SMILES
        (usually input sequnce).
        measures_to_log: Dictonary with values to log.
        emb_activation: Activation function used in the bottleneck layer.
        lr: Learning rate for training the model.
        lr_decay: Boolean to define if learning rate decay is used.
        lr_decay_frequency: Number of steps between learning rate decay steps.
        lr_decay_factor: Amount of learning rate decay.
        beam_width: Width of the the window used for the beam search decoder.
    """

    def __init__(self, mode, iterator, hparams):
        """Constructor for base translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        self.mode = mode
        self.iterator = iterator
        self.embedding_size = hparams.emb_size
        self.encode_vocabulary = {
            v: k for k, v in np.load(hparams.encode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.encode_voc_size = len(self.encode_vocabulary)
        self.decode_vocabulary = {
            v: k for k, v in np.load(hparams.decode_vocabulary_file, allow_pickle=True).item().items()
        }
        self.decode_vocabulary_reverse = {v: k for k, v in self.decode_vocabulary.items()}
        self.decode_voc_size = len(self.decode_vocabulary)
        self.one_hot_embedding = hparams.one_hot_embedding
        self.char_embedding_size = hparams.char_embedding_size
        
        ### TF GET VAR, N.B. Initializer
        #self.global_step = tf.get_variable('global_step',
        #                                   [],
        #                                   ### CONSTANT INITIALIZER
        #                                   initializer=tf.constant_initializer(0),
        #                                   trainable=False)
        self.global_step = torch.tensor(0, requires_grad = False)


        self.save_dir = hparams.save_dir
        ### Check this is in line with model saver
        self.checkpoint_path = os.path.join(self.save_dir, 'model.ckpt')
        self.batch_size = hparams.batch_size
        self.rand_input_swap = hparams.rand_input_swap
        self.measures_to_log = {}
        if hparams.emb_activation == "tanh":
            self.emb_activation = F.tanh ### Changed
        elif hparams.emb_activation == "linear":
            self.emb_activation = lambda x: x
        else:
            raise ValueError("This activation function is not implemented...")
        if mode == "TRAIN":
            ### Potentially subject to change
            self.lr = hparams.lr
            self.lr_decay = hparams.lr_decay
            self.lr_decay_frequency = hparams.lr_decay_frequency
            self.lr_decay_factor = hparams.lr_decay_factor
        if mode == "DECODE":
            self.beam_width = hparams.beam_width
        if mode not in ["TRAIN", "EVAL", "ENCODE", "DECODE"]:
            raise ValueError("Choose one of following modes: TRAIN, EVAL, ENCODE, DECODE")
        
        ### TORCH LAYERS

    ### TO EVENTUALLY BE RENAMED/MOVED,  becomes the forward
    ### def forward(self):
    def build_graph(self):
        """Method that defines the graph for a translation model instance."""
        if self.mode in ["TRAIN", "EVAL"]:
            ### NAME SCOPE
            #with tf.name_scope("Input"):
            (self.input_seq,
            self.shifted_target_seq,
            self.input_len,
            self.shifted_target_len,
            self.target_mask,
            encoder_emb_inp,
            decoder_emb_inp) = self._input()
            ### VAR SCOPE
            #with tf.variable_scope("Encoder"):
            encoded_seq = self._encoder(encoder_emb_inp)
            ### VAR SCOPE
            #with tf.variable_scope("Decoder"):
            logits = self._decoder(encoded_seq, decoder_emb_inp)
                ### ARGMAX, N.B. int32
                ### self.prediction = tf.argmax(logits, axis=2, output_type=tf.int32)
            self.prediction = torch.argmax(logits, dim=2)
            ### NAME SCOPE
            ### with tf.name_scope("Measures"):
            self.loss = self._compute_loss(logits)
            self.accuracy = self._compute_accuracy(self.prediction)
            self.measures_to_log["loss"] = self.loss
            self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                ### NAME SCOPE
                #with tf.name_scope("Training"):
                self._training()

        if self.mode == "ENCODE":
            ### NAME SCOPE
            #with tf.name_scope("Input"):
            ### PLACEHOLDER, N.B. int32 ---> Directly use in forward
            self.input_seq = tf.placeholder(tf.int32, [None, None])
            ### PLACEHOLDER, N.B. int32 ---> Directly use in forward
            self.input_len = tf.placeholder(tf.int32, [None])
            encoder_emb_inp = self._emb_lookup(self.input_seq)
            ### VAR SCOPE
            #with tf.variable_scope("Encoder"):
            self.encoded_seq = self._encoder(encoder_emb_inp)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                ### ONE HOT, indices, depth
                ### Takes [0,N] interval, depth N
                #self.decoder_embedding = tf.one_hot(
                #    list(range(0, self.decode_voc_size)),
                #    self.decode_voc_size
                #)
                self.decoder_embedding = F.one_hot(
                    torch.arange(0, self.decode_voc_size), 
                    num_classes = self.decode_voc_size 
                )



            elif self.encode_vocabulary == self.decode_vocabulary:
                ### GET VARIABLE
                self.decoder_embedding = tf.get_variable(
                    "char_embedding",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                ### Figure out how to initialise the tensors consistently
                self.decoder_embedding = torch.tensor(...init, requires_grad = True)
            else:
                ### GET VARIABLE
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                self.decoder_embedding = torch.tensor(...init, required_grad = True)
            ### NAME SCOPE
            #with tf.name_scope("Input"):
            ### PLACEHOLDER, float
            self.encoded_seq = tf.placeholder(tf.float32,
                                              [None, self.embedding_size])
            ### PLACEHOLDER int
            self.maximum_iterations = tf.placeholder(tf.int32, [])
            ### PLACEHOLDER int
            self.maximum_iterations = tf.placeholder(tf.int32, [])

            ### VAR SCOPE
            # with tf.variable_scope("Decoder"):
            self.output_ids = self._decoder(self.encoded_seq)

        ### TF TRAIN SAVER
        self.saver_op = tf.train.Saver()
        self.saver_op = torch.save ... ## Needs more thought

    def _input(self, with_features=False):
        """Method that defines input part of the graph for a translation model instance.

        Args:
            with_features: Defines if in addition to input and output sequnce futher
            molecular features e.g. logP are expected from the input pipleine iterator.
        Returns:
            input_seq: The input sequnce.
            shifted_target_seq: The target sequnce shifted by one charcater to the left.
            input_len: Number of tokens in input.
            shifted_target_len: Number of tokens in the shifted target sequence.
            target_mask: shifted target sequence with masked padding tokens.
            encoder_emb_inp: Embedded input sequnce (contineous character embedding).
            decoder_emb_inp: Embedded input sequnce (contineous character embedding).
            mol_features: if Arg with_features is set to True, the molecular features of the
            input pipleine are passed.
        """
        ### TF DEVICE --> CARE
        with tf.device('/cpu:0'):
            if with_features:
                seq1, seq2, seq1_len, seq2_len, mol_features = self.iterator.get_next()
            else:
                seq1, seq2, seq1_len, seq2_len = self.iterator.get_next()
            if self.rand_input_swap:
                ### TF RANDOM UNIFORM --> MIGHT BE BETTER REPHRASED with a not?
                #rand_val = tf.random_uniform([], dtype=tf.float32)
                rand_val = torch.rand([], dtype = torch.float32)
                ### TF COND
                #input_seq = tf.cond(tf.greater_equal(rand_val, 0.5),
                #                    lambda: seq1, lambda: seq2)
                input_seq = torch.where(rand_val >= 0.5, seq1, seq2)
                ### TF COND
                #input_len = tf.cond(tf.greater_equal(rand_val, 0.5),
                #                    lambda: seq1_len, lambda: seq2_len)
                input_len = torch.where(rand_val >= 0.5, seq1_len, seq2_len)
            else:
                input_seq = seq1
                input_len = seq1_len
            target_seq = seq2
            target_len = seq2_len
            ### TF RESHAPE / TF SHAPE
            shifted_target_len = tf.reshape(target_len, [tf.shape(target_len)[0]]) - 1
            ### TF SLICE
            shifted_target_seq = tf.slice(target_seq, [0, 1], [-1, -1])
            ### TF SEQUENCE MASK
            target_mask = tf.sequence_mask(shifted_target_len, dtype=tf.float32)
            ### TF REDUCE SUM
            target_mask = target_mask / tf.reduce_sum(target_mask)
            ### TF RESHAPE/ TF SHAPE
            input_len = tf.reshape(input_len, [tf.shape(input_len)[0]])

        encoder_emb_inp, decoder_emb_inp = self._emb_lookup(input_seq, target_seq)
        if with_features:
            return (input_seq, shifted_target_seq, input_len, shifted_target_len,
                    target_mask, encoder_emb_inp, decoder_emb_inp, mol_features)
        else:
            return (input_seq, shifted_target_seq, input_len, shifted_target_len,
                    target_mask, encoder_emb_inp, decoder_emb_inp)

    def _emb_lookup(self, input_seq, target_seq=None):
        """Method that performs an embedding lookup to embed the one-hot encoded input
        and output sequnce into the trainable contineous character embedding.

        Args:
            input_seq: The input sequnce.
            target_seq: The target sequnce.
        Returns:
            encoder_emb_inp: Embedded input sequnce (contineous character embedding).
            decoder_emb_inp: Embedded input sequnce (contineous character embedding).
        """
        if self.one_hot_embedding:
            ### TF ONE HOT
            #self.encoder_embedding = tf.one_hot(
            #    list(range(0, self.encode_voc_size)),
            #    self.encode_voc_size
            #)
            self.encoder_embedding = F.one_hot(
                torch.arange(0, self.encode_voc_size), 
                num_classes = self.encode_voc_size 
            )
        else:
            ### TF GET VAR
            self.encoder_embedding = tf.get_variable(
                "char_embedding",
                [self.encode_voc_size, self.char_embedding_size]
            )
            self.encoder_embedding = torch,tensor(...init, required_grad = True)

        ### TF.nn EMBEDDING LOOKUP
        ## encoder_emb_inp = tf.nn.embedding_lookup(self.encoder_embedding, input_seq)
        encoder_emb_inp = torch.gather(self.encoder_embedding, ?, input_seq) ## Test this carefully

        if self.mode != "ENCODE":
            assert target_seq is not None
            if self.encode_vocabulary == self.decode_vocabulary:
                self.decoder_embedding = self.encoder_embedding
            elif self.one_hot_embedding:
                ## TF ONE HOT
                #self.decoder_embedding = tf.one_hot(
                #    list(range(0, self.decode_voc_size)),
                #    self.decode_voc_size
                #)
                self.decoder_embedding = F.one_hot(
                    torch.arange(0, self.decode_voc_size), 
                    num_classes = self.decode_voc_size 
                )
            else:
                ## TF GET VAR
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                self.decoder_embedding = torch.tensor(...init, requires_grad = True)
            ## TF nn EMBEDDING LOOKUP
            #decoder_emb_inp = tf.nn.embedding_lookup(self.decoder_embedding, target_seq)
            decoder_emb_inp = torch.gather(self.decoderembedding, dim = ? , target_seq)
            return encoder_emb_inp, decoder_emb_inp
        else:
            return encoder_emb_inp

    def _training(self):
        """Method that defines the training operation of the training models graph."""

        if self.lr_decay:
            ## TF TRAIN EXPONENTIAL DECAY
            #self.lr = tf.train.exponential_decay(self.lr,
            #                                     self.global_step,
            #                                     self.lr_decay_frequency,
            #                                     self.lr_decay_factor,
            #                                     staircase=True,)


            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma =)
            ## Here gamma should be multiplicative factor of decay, i.e. lr_decay_factor
            ## Order needs to be fixed, i.e. rewrite to be more torch-like

        ### TF TRAIN ADAM OPT
        #self.opt = tf.train.AdamOptimizer(self.lr, name='optimizer')
        self.opt = torch.optim.Adam(self.parameters(),lr = self.lr)  ### BUT, self.parameters() probably not defined, see ODO

        ### TF --> IMPLICIT, compute_gradients method
        grads = self.opt.compute_gradients(self.loss)

        ### TF CLIP_BY_VALUE
        grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
        self.train_step = self.opt.apply_gradients(grads, self.global_step)

    @abstractmethod
    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        raise NotImplementedError("Must override _encoder in child class")

    @abstractmethod
    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        raise NotImplementedError("Must override _decoder in child class")

    def _compute_loss(self, logits):
        """Method that calculates the loss function."""
        ### TF.nn SPARSE SOFTMAX CROSS ENTROPY WITH LOGITS
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #   labels=self.shifted_target_seq,
        #  logits=logits)

        crossent = F.cross_entropy(logits, self.shifted_target_seq) ### set x = logits, target = self.shifted...

        ### TF REDUCE SUM
        #loss = (tf.reduce_sum(crossent * self.target_mask))
        loss = torch.sum(crossent * self.target_mask)
        return loss

    def _compute_accuracy(self, prediction):
        """Method that calculates the character-wise translation accuracy."""
        ### TF CAST, EQUAL
        #right_predictions = tf.cast(tf.equal(prediction, self.shifted_target_seq), tf.float32)
        right_predictions = torch.eq(prediction, self.shifted_target_seq) ##.double()
        ### TF REDUCE SUM
        #accuracy = (tf.reduce_sum(right_predictions * self.target_mask))
        accuracy = torch.sum(right_predictions * self.target_mask)
        return accuracy

    def train(self, sess):
        """Method that can be called to perform a training step.

        Args:
            sess: The Session the model is running in.
        Returns:
            step: The global step.
        """
        assert self.mode == "TRAIN"
        ### TF SESSION + sess.run alternative required
        #_, step = sess.run([self.train_step, self.global_step])
        ???
        return step

    def eval(self, sess):
        """Method that can be called to perform a evaluation step.

        Args:
            sess: The Session the model is running in.
        Returns:
            step: The loged measures.
        """
        ### TF SESSION + sess.run
        return sess.run(list(self.measures_to_log.values()))

    def idx_to_char(self, seq):
        """Helper function to transform the one-hot encoded sequnce tensor back to string-sequence.

        Args:
            seq: sequnce of one-hot encoded characters.
        Returns:
            string sequnce.
        """
        return ''.join([self.decode_vocabulary_reverse[idx] for idx in seq
                        if idx not in [-1, self.decode_vocabulary["</s>"],
                                       self.decode_vocabulary["<s>"]]])

    def seq2emb(self, sess, input_seq, input_len):
        """Method to run a forwards path up to the bottleneck layer (ENCODER).
        Encodes a one-hot encoded input sequnce.

        Args:
            sess: The Session the model is running in.
            input_seq: sequnces of one-hot encoded characters.
            input_len: number of characters per sequnce.
        Returns:
            Embedding of the input sequnces.
        """
        assert self.mode == "ENCODE"
        ### TF SESSION + sess.run
        return sess.run(self.encoded_seq, {self.input_seq: input_seq,
                                           self.input_len: input_len})
        ### HERE COME UP WITH model(x) type expression to call encoder.

    def emb2seq(self, sess, embedding, num_top, maximum_iterations=1000):
        """Method to run a forwards path from bottleneck layer to output sequnce (DECODER).
        Decodes the embedding (molecular descriptor) back to a sequnce representaion.

        Args:
            sess: The Session the model is running in.
            embedding: Embeddings (molecular descriptors) of the input sequnces.
            num_top: Number of most probable sequnces as output of the beam search decoder
        Returns:
            Embedding of the input sequnces.
        """
        assert self.mode == "DECODE"
        ### TF SESSION + sess.run
        output_seq = sess.run(self.output_ids, {self.encoded_seq: embedding,
                                                self.maximum_iterations: maximum_iterations})
        return [[self.idx_to_char(seq[:, i]) for i in range(num_top)] for seq in output_seq]

    def initilize(self, sess, overwrite_saves=False):
        """Function to initialize variables in the model graph and creation of save folder.

        Args:
            sess: The Session the model is running in.
            overwrite_saves: Defines whether to overwrite the files (recreate directory) if a folder
            with same save file path exists.
        Returns:
            step: Initial value of global step.
        """
        assert self.mode == "TRAIN"
        ### TF SESSION + sess.run
        sess.run(tf.global_variables_initializer())
        ### HERE CHECK any required init. of pytorch methods


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print('Create save file in: ', self.save_dir)
        elif overwrite_saves:
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
        else:
            raise ValueError("Save directory %s already exist." %(self.save_dir))
        return sess.run(self.global_step)

    def restore(self, sess, restore_path=None):
        """ Helper Function to restore the variables in the model graph."""

        if restore_path is None:
            restore_path = self.checkpoint_path
        ### SELF SAVER OP --> .restore
        self.saver_op.restore(sess, restore_path)

        ### Fully replace as neccesary

        if self.mode == "TRAIN":
            ### TF SESSION + session.run
            step = sess.run(self.global_step)
            print("Restarting training at step %d" %(step))
            return step

    def save(self, sess):
        """Wrapper function save model to file."""
        ### SELF SAVER OP --> .save
        self.saver_op.save(sess, self.checkpoint_path)
        #torch.save(self.state_dict, '...')

class GRUSeq2Seq(BaseModel):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with Gate Recurrent Units (GRUs). Encoder and Decoder architecture are
    the same.

    Attributes:
        cell_size: list defining the number of Units in each GRU cell.
        reverse_decoding: whether to invert the cell_size list for the Decoder.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the GRU translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size
        self.reverse_decoding = hparams.reverse_decoding
        
        ### TORCH LAYERS


        self.dense_layer = nn.Linear(<in>,self.embedding_size, device=device)

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        ### TF NN RNN_CELL GRU
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]

        torch.nn.GRU( ...

        ### TF CONTRIB RNN MultiRNNCell < list
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)

        ### ?? Consider REINVENT class MultiGRU... 

        ### TF.nn.dynamic_rnn, NB float
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        ### TF LAYERS DENSE, TF CONCAT
        #emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
        #                      self.embedding_size,
        #                      activation=self.emb_activation
        #                     )

        emb = self.emb_activation(self.dense_layer(torch.cat(encoder_state, dim = 1)))

        return emb

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        if self.reverse_decoding:
            self.cell_size = self.cell_size[::-1]
        ### TF NN RNN_CELL GRUCell
        decoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        ### TF CONTRIB RNN MultiRNNCell
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)
        ### TF LAYERS DENSE
        decoder_cell_inital = tf.layers.dense(encoded_seq, sum(self.cell_size))
        ### TF SPLIT
        decoder_cell_inital = tuple(tf.split(decoder_cell_inital, self.cell_size, 1))
        ### TF LAYERS DENSE
        projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
        if self.mode != "DECODE":
            ### TF CONTRIB SEQ2SEQ TRAINING HELPER
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                       sequence_length=self.shifted_target_len,
                                                       time_major=False)
            ### TF CONTRIB SEQ2SEQ BASIC DECODER
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_cell_inital,
                                                      output_layer=projection_layer)
            ### TF CONTRIB SEQ2SEQ DYNAMIC DECODE --> CHECK IMPUTE FINISHED
            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                         impute_finished=True,
                                                                         output_time_major=False)
            return outputs.rnn_output
        else:
            ### TF CONTRIB SEQ2SEQ TILE BATCH
            decoder_cell_inital = tf.contrib.seq2seq.tile_batch(decoder_cell_inital,
                                                                self.beam_width)
            ### TF FILL
            start_tokens = tf.fill([tf.shape(encoded_seq)[0]], self.decode_vocabulary['<s>'])
            end_token = self.decode_vocabulary['</s>']
            ### TF CONTRIB SEQ2SEQ BEAM SEARCH Decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.decoder_embedding,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=decoder_cell_inital,
                beam_width=self.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)

            ### TF CONTRIB SEQ2SEQ DYNAMIC_DECODE
            outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False,
                output_time_major=False,
                maximum_iterations=self.maximum_iterations
            )

            return outputs.predicted_ids
        
class GRUVAE(GRUSeq2Seq):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        self.div_loss_scale = hparams.div_loss_scale
        self.div_loss_rate = hparams.div_loss_rate
        
        ### TORCH LAYERS
        self.MultiRNN =

        self.dense_layer_1 = nn.Linear(<in>,self.embedding_size,...)
        self.dense_layer_2 = nn.Linear(<in>,self.embedding_size,...)
        
    def _encoder(self, encoder_emb_inp):

        """Method that defines the encoder part of the translation model graph."""
        ###
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        ###
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        ###
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        ###
        #loc = tf.layers.dense(tf.concat(encoder_state, axis=1),
        #                      self.embedding_size
        #                     )
        loc = self.dense_layer_1(torch.cat(encoder_state, dim = 1))
        ###
        #log_scale = tf.layers.dense(tf.concat(encoder_state, axis=1),
        #                        self.embedding_size
        #                       )
        log_scale = self.dense_layer_2(torch.cat(encoder_state, dim = 1))
        return loc, log_scale
    
    def _sampler(self, loc, log_scale):
        ### TF RANDOM NORMAL
        #epsilon = tf.random_normal(
        #    ### TF SHAPE
        #    shape=[tf.shape(loc)[0], self.embedding_size],
        #    mean=0,
        #    stddev=1
        #)
        epsilon = torch.normal(mean = torch.zeros( loc.size()[0], self.embedding_size), std = 1)

        ### TF.EXP
        return loc + torch.exp(log_scale) * epsilon
    
    def _compute_loss(self, logits, loc, log_scale):
        """Method that calculates the loss function."""
        ### TF SSCEWL
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    labels=self.shifted_target_seq,
        #    logits=logits)
        crossent = F.crossent(logits, self.shifted_target_seq)  ## Note swapped arguments
        ### TF REDUCE SUM
        #crossent = tf.reduce_sum(crossent * self.target_mask, axis=1)
        crossent = torch.sum( crossent * self.target_mask, dim = 1)


        ### DIVERGENCE
        #divergence = -0.5 * tf.reduce_sum(1 + 2*log_scale - tf.square(loc) - tf.square(tf.exp(log_scale)), axis=-1)
        divergence = -0.5 * torch.sum(1 + 2*log_scale - torch.square(loc) - torch.square(torch.exp(log_scale)), dim = -1) ## Check dim = -1 behaviour

        ### TF REDUCE MEAN
        #self.measures_to_log["crossent"] = tf.reduce_mean(crossent)
        self.measures_to_log["crossent"] = torch.mean(crossent)
        ### TF REDUCE MEAN
        #self.measures_to_log["divergence"] = tf.reduce_mean(divergence)
        self.measures_to_log["divergence"] = torch.mean(divergence)
        
        ### TF TRAIN EXPONENTIAL DECAY (figure out reshuffle of lr)
        div_loss_scale = self.div_loss_scale - tf.train.exponential_decay(self.div_loss_scale,
                                                 self.global_step,
                                                 10000,
                                                 self.div_loss_rate,
                                                 staircase=True,)
        ###

        self.measures_to_log["div_loss_scale"] = div_loss_scale
        ###
        #return tf.reduce_mean(crossent + div_loss_scale * divergence)
        return torch.mean(crossent + div_loss_scale * divergence)
    
    def build_graph(self):
        """Method that defines the graph for a translation model instance."""
        if self.mode in ["TRAIN", "EVAL"]:
            ###
            #with tf.name_scope("Input"):
            (self.input_seq,
            self.shifted_target_seq,
            self.input_len,
            self.shifted_target_len,
            self.target_mask,
            encoder_emb_inp,
            decoder_emb_inp) = self._input()
            ###
            #with tf.variable_scope("Encoder"):
            loc, log_scale = self._encoder(encoder_emb_inp)
            encoded_seq = self._sampler(loc, log_scale)
            ###
            #with tf.variable_scope("Decoder"):
            logits = self._decoder(encoded_seq, decoder_emb_inp)
                ### TF ARGMAX
                ### self.prediction = tf.argmax(logits, axis=2, output_type=tf.int32)
            self.prediction = torch.argmax(logits, dim=2)
            ###
            #with tf.name_scope("Measures"):
                #crossent, divergence, self.loss = self._compute_loss(logits, posterior)
            self.loss = self._compute_loss(logits, loc, log_scale)
            self.accuracy = self._compute_accuracy(self.prediction)
            self.measures_to_log["loss"] = self.loss
            self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                ###
                #with tf.name_scope("Training"):
                self._training()

        if self.mode == "ENCODE":
            ###
            #with tf.name_scope("Input"):
            self.input_seq = tf.placeholder(tf.int32, [None, None])
            self.input_len = tf.placeholder(tf.int32, [None])
            encoder_emb_inp = self._emb_lookup(self.input_seq)
            ###
            #with tf.variable_scope("Encoder"):
            loc, log_scale = self._encoder(encoder_emb_inp)
            self.encoded_seq = self._sampler(loc, log_scale)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                ### TF ONE HOT
                #self.decoder_embedding = tf.one_hot(
                #    list(range(0, self.decode_voc_size)),
                #    self.decode_voc_size
                #)
                self.decoder_embedding = F.one_hot(
                    torch.arange(0, self.decode_voc_size), 
                    num_classes = self.decode_voc_size 
                )
            elif self.encode_vocabulary == self.decode_vocabulary:
                ### TF GET VAR
                #self.decoder_embedding = tf.get_variable(
                #    "char_embedding",
                #    [self.decode_voc_size, self.char_embedding_size]
                #)
                self.decoder_embedding = torch.tensor(...init, requires_grad = True)
            else:
                ### TF GET VAR
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                self.decoder_embedding = torch.tensor(...init, requires_grad = True)

            #with tf.name_scope("Input"):
            ### TF PLACEHOLDER
            self.encoded_seq = tf.placeholder(tf.float32,
                                                  [None, self.embedding_size])

            ### TF VAR SCOPE
            #with tf.variable_scope("Decoder"):
            self.output_ids = self._decoder(self.encoded_seq)

        ### TF.train.SAVER
        self.saver_op = tf.train.Saver()
        self.saver_op = torch.save() ### Needs more thought

class NoisyGRUSeq2Seq(GRUSeq2Seq):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder and
    Decoder with Gate Recurrent Units (GRUs) with input dropout and a Gaussian Noise term
    after the bottleneck layer. Encoder and Decoder architecutre are the same.

    Attributes:
        input_dropout: Dropout rate of a Dropout layer after the character embedding of the
        input sequnce.
        emb_noise: Standard deviation of the Gaussian Noise term after the bottleneck layer.
    """

    def __init__(self, mode, iterator, hparams):
        """Constructor for the Noisy GRU translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise
        
        ### TORCH LAYERS
        self.dropout_layer = 
        self.dense_layer = nn.Linear(<in>,self.embedding_size,...) 

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        if (self.mode == "TRAIN") & (self.input_dropout > 0.0):
            ### TF.shape
            #max_time = tf.shape(encoder_emb_inp)[1]
            max_time = torch.size(encoder_emb_inp)[1]
            ### TF DROPOUT LAYER
            encoder_emb_inp = tf.nn.dropout(encoder_emb_inp,
                                            1. - self.input_dropout,
                                            noise_shape=[self.batch_size, max_time, 1])
        
        ### TF GRU CELL
        encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        ### TF MULTI RNN
        encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        ### TF DYNAMIC RNN
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                           encoder_emb_inp,
                                                           sequence_length=self.input_len,
                                                           dtype=tf.float32,
                                                           time_major=False)
        ### TF DENSE
        #emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
        #                      self.embedding_size
        #                     )
        emb = self.dense_layer(torch.cat(encoder_state, dim = 1))

        if (self.mode == "TRAIN") & (self.emb_noise > 0.0):
            ### TF RANDOM NORMAL
            #emb += tf.random_normal(shape=tf.shape(emb),
            #                        mean=0.0,
            #                        stddev=self.emb_noise,
            #                        dtype=tf.float32)
            emb += torch.normal(mean = torch.zeros(torch.size(emb)), std = self.emb_noise)
        emb = self.emb_activation(emb)
        return emb

class LSTMSeq2Seq(BaseModel):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with Long short-term memory units (LSTM). Encoder and Decoder architecutre
    are the same.

    Attribures:
        cell_size: list defining the number of Units in each GRU cell.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the LSTM translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.cell_size = hparams.cell_size
        
        ### TORCH LAYERS
        self.encoder_LSTM = torch.nn.LSTM(input_size = , hidden_size =, num_layers =)
        self.decoder_LSTM = torch.nn.LSTM(input_size = , hidden_size =, num_layers =)

        self.encoder_dense_layer = nn.Linear(<in>,self.embedding_size)

        self.decoder_projection = nn.Linear(<in>, self.decode_voc_size, use_boa = False)


    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        ### TF LSTM/MULTIRNNN
        #encoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        #encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        #encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
        #                                                   encoder_emb_inp,
        #                                                   sequence_length=self.input_len,
        #                                                   dtype=tf.float32,
        #                                                   time_major=False)   ### thus IN/OUT shape = [batch_size, max_time, depth]
        #encoder_state_c = [state.c for state in encoder_state]

        ### EXAMPLE TORCH
        #>>> rnn = nn.LSTM(10, 20, 2)
        #>>> input = torch.randn(5, 3, 10)
        #>>> h0 = torch.randn(2, 3, 20)
        #>>> c0 = torch.randn(2, 3, 20)
        #>>> output, (hn, cn) = rnn(input, (h0, c0))
        h0 = 
        c0 =
        encoder_outputs, encoder_state = self.encoder_LSTM(encoder_emb_inp, (h0, c0))
        encoder_state_c = encoder_state[1]
        ### TF DENSE LAYER
        #emb = tf.layers.dense(tf.concat(encoder_state_c, axis=1),
        #                      self.embedding_size,
        #                      activation=self.emb_activation
        #                     )
        emb = self.emb_activation(self.encoder_dense_layer(torch.cat(encoder_state_c, dim = 1)))
        return emb

    def _decoder(self, encoded_seq, decoder_emb_inp=None):
        """Method that defines the decoder part of the translation model graph."""
        ### TF LSTM/MULTIRNN
        decoder_cell = [tf.nn.rnn_cell.LSTMCell(size) for size in self.cell_size]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell)

        ### TF DENSE LAYER
        initial_state_c_full = tf.layers.dense(encoded_seq, sum(self.cell_size))
        ### TF SPLIT
        initial_state_c = tuple(tf.split(initial_state_c_full, self.cell_size, 1))
        ### TF ZEROS LIKE
        initial_state_h_full = tf.zeros_like(initial_state_c_full)
        ### TF SPLIT
        initial_state_h = tuple(tf.split(initial_state_h_full, self.cell_size, 1))
        ### LSTM
        decoder_cell_inital = tuple(
            [tf.contrib.rnn.LSTMStateTuple(
                initial_state_c[i],
                initial_state_h[i]) for i in range(len(self.cell_size))
            ]
        )
        ### TF TRAINING HELPER
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                   sequence_length=self.shifted_target_len,
                                                   time_major=False)
        ### TF DENSE 'projection_layer'
        projection_layer = tf.layers.Dense(self.decode_voc_size, use_bias=False)
        ## CHECK HOW THIS WORKS WITH THE OTHERSprojection_layer = self.decoder_projection(...) ##
        ### TF DECODER
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  helper,
                                                  decoder_cell_inital,
                                                  output_layer=projection_layer)
        ### TF DYNAMIC DECODER
        outputs, output_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                     impute_finished=True,
                                                                     output_time_major=False)
        return outputs.rnn_output

class Conv2GRUSeq2Seq(GRUSeq2Seq):
    """Translation model class with a multi-layer 1-D Convolutional Neural Network as Encoder.
    The Decoder is still a RNN with GRU cells.

    Attributes:
        conv_hidden_size: List defining the number of filters in each layer.
        kernel_size: List defining the width of the 1-D conv-filters in each layer.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the Convolutional translation model class.

        Args:
            mode: The mode the model is supposed to run (e.g. TRAIN, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not TRAIN, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.conv_hidden_size = hparams.conv_hidden_size
        self.kernel_size = hparams.kernel_size

        ### TORCH LAYERS
        encoder_conv1d = {}
        encoder_max_pool = {}
        for i, size in enumerate(self.conv_hidden_size):
            encoder_conv1d[i] = nn.Conv1d(<in>,<out>,self.kernel_size[i],padding_mode = 'SAME')
            encoder_max_pool[i] = torch.
        encoder[i??+1] = torch.


        encoder_dense_layer = torch.

    def _encoder(self, encoder_emb_inp):
        ### CHECK USE OF THE CONV1D LAYER
        """Method that defines the encoder part of the translation model graph."""
        for i, size in enumerate(self.conv_hidden_size):
            ### TF LAYERS CONV1D
            x = tf.layers.conv1d(encoder_emb_inp,
                                 size,
                                 self.kernel_size[i],
                                 activation=tf.nn.relu,
                                 padding='SAME')
            if i+1 < len(self.conv_hidden_size):
                ### TF LAYERS MAX POOLING1D
                x = tf.layers.max_pooling1d(x, 3, 2, padding='SAME')

        ### TF LAYERS CONV1D
        x = tf.layers.conv1d(x,
                             self.conv_hidden_size[-1],
                             1,
                             activation=tf.nn.relu,
                             padding='SAME')

        ### TF LAYERS DENSE, REDUCE MEAN
        emb = tf.layers.dense(tf.reduce_mean(x, axis=1),
                              self.embedding_size,
                              activation=self.emb_activation
                             )
        emb = encoder_dense_layer(torch.mean(x, dim = 1))
        emb = self.emb_activation(emb) ## Check
        return emb

class GRUSeq2SeqWithFeatures(GRUSeq2Seq):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder
    and Decoder with  Gate Recurrent Units (GRUs) with an additional feature classification
    task. Encoder and Decoder architecutre are the same.

    Attribures:
        num_features: Number of features to prediced.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the GRU translation model with feature classification class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.num_features = hparams.num_features
        
        ### TORCH LAYERS
        self.feature_regression_dense_1 = nn.Linear(<in>,512,...)
        self.feature_regression_dense_2 = nn.Linear(512,128,...)
        self.feature_regression_dense_2 = nn.Linear(128,self.num_features,...)
        self.features_MSE_loss = nn.MSELoss()

    def build_graph(self):
        """Method that defines the graph for a translation model instance with the additional
        feature prediction task.
        """
        if self.mode in ["TRAIN", "EVAL"]:
            #with tf.name_scope("Input"):
            (self.input_seq,
            self.shifted_target_seq,
            self.input_len,
            self.shifted_target_len,
            self.target_mask,
            encoder_emb_inp,
            decoder_emb_inp,
            self.mol_features) = self._input(with_features=True)

            #with tf.variable_scope("Encoder"):
            encoded_seq = self._encoder(encoder_emb_inp)

            #with tf.variable_scope("Decoder"):
            sequence_logits = self._decoder(encoded_seq, decoder_emb_inp)
                ### TF ARGMAX
                #self.sequence_prediction = tf.argmax(sequence_logits,
                #                                     axis=2,
                #                                     output_type=tf.int32)
            self.sequence_prediction = torch.argmax(sequence_logits, dim=2)

            #with tf.variable_scope("Feature_Regression"):
            feature_predictions = self._feature_regression(encoded_seq)

            #with tf.name_scope("Measures"):
            self.loss_sequence, self.loss_features = self._compute_loss(sequence_logits,
                                                                            feature_predictions)
            self.loss = self.loss_sequence + self.loss_features
            self.accuracy = self._compute_accuracy(self.sequence_prediction)
            self.measures_to_log["loss"] = self.loss
            self.measures_to_log["accuracy"] = self.accuracy

            if self.mode == "TRAIN":
                #with tf.name_scope("Training"):
                self._training()

        if self.mode == "ENCODE":
            ### TF NAME SCOPE
            #with tf.name_scope("Input"):
            ### TF PLACEHOLDER
            self.input_seq = tf.placeholder(tf.int32, [None, None])
            ### TF PLACEHOLDER
            self.input_len = tf.placeholder(tf.int32, [None])
            encoder_emb_inp = self._emb_lookup(self.input_seq)

            #with tf.variable_scope("Encoder"):
            self.encoded_seq = self._encoder(encoder_emb_inp)

        if self.mode == "DECODE":
            if self.one_hot_embedding:
                ### TF Decoder
                #self.decoder_embedding = tf.one_hot(
                #    list(range(0, self.decode_voc_size)),
                #    self.decode_voc_size
                #)
                self.decoder_embedding = F.one_hot(
                    torch.arange(0, self.decode_voc_size), 
                    num_classes = self.decode_voc_size 
                )
            elif self.encode_vocabulary == self.decode_vocabulary:
                ### TF GET VAR
                self.decoder_embedding = tf.get_variable(
                    "char_embedding",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                self.decoder_embedding = torch.tensor(..., requires_grad = True)

            else:
                ### TF GET VAR
                self.decoder_embedding = tf.get_variable(
                    "char_embedding2",
                    [self.decode_voc_size, self.char_embedding_size]
                )
                self.decoder_embedding = toach.tensor(..., requires_grad = True)

            ### TF NAME SCOPE
            #with tf.name_scope("Input"):
            ### TF PLACEHOLDER
            self.encoded_seq = tf.placeholder(tf.float32, [None, self.embedding_size])
            ### TF PLACEHOLDER
            self.maximum_iterations = tf.placeholder(tf.int32, [])
            ### TF NAME SCOPE
            #with tf.variable_scope("Decoder"):
            self.output_ids = self._decoder(self.encoded_seq)
        
        ### TF SAVER
        self.saver_op = tf.train.Saver()
        ### Probably needs to be triggered somewhere else
        torch.save()

    def _feature_regression(self, encoded_seq):
        """Method that defines the feature classification part of the graph."""
        ## TF LAYERS DENSE
        #x = tf.layers.dense(inputs=encoded_seq,
        #                    units=512,
        #                    activation=tf.nn.relu
        #                    )
        ## TF LAYERS DENSE
        #x = tf.layers.dense(inputs=x,
        #                    units=128,
        #                    activation=tf.nn.relu
        #                    )
        ## TF LAYERS DENSE
        #x = tf.layers.dense(inputs=x,
        #                    units=self.num_features,
        #                    activation=None
        #                    )
        x = F.relu(self.feature_regression_dense_1(encoded_seq))
        x = F.relu(self.feature_regression_dense_2(x))
        x = self.feature_regression_dense_2(x)
        return x

    def _compute_loss(self, sequence_logits, features_predictions):
        """Method that calculates the loss function."""
        ## TF SPARSE SOFTMAX
        #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.shifted_target_seq,
        #                                                          logits=sequence_logits)
        crossent = F.crossent(sequence_logits, self.shifted_target_seq)
        ### TF REDUCE MEAN
        #loss_sequence = (tf.reduce_sum(crossent * self.target_mask))
        loss_sequence = torch.sum(crossent * self.target_mask)
        ### TF LOSSES MSE
        #loss_features = tf.losses.mean_squared_error(labels=self.mol_features,
        #                                             predictions=features_predictions,
        #                                            )
        loss_features = self.features_MSELoss(features_predictions, self.mol_features)
        ### Example: loss_features.backward()
        return loss_sequence, loss_features

class NoisyGRUSeq2SeqWithFeatures(GRUSeq2SeqWithFeatures):
    """Translation model class with a multi-layer Recurrent Neural Network as Encoder and Decoder
    with Gate Recurrent Units (GRUs) with input dropout and a Gaussian Noise Term after the
    bottleneck layer and an additional feature classification task. Encoder and Decoder architecture
    are the same.

    Attribures:
        input_dropout: Dropout rate of a Dropout layer after the character embedding of the input
        sequnce.
        emb_noise: Standard deviation of the Gaussian Noise term after the bottleneck layer.
    """
    def __init__(self, mode, iterator, hparams):
        """Constructor for the Noisy GRU translation model with feature classification class.

        Args:
            mode: The mode the model is supposed to run (e.g. Train, EVAL, ENCODE, DECODE).
            iterator: The iterator of the input pipeline.
            hparams: Hyperparameters defined in file or flags.
        Returns:
            None
        Raises:
            ValueError: if mode is not Train, EVAL, ENCODE, DECODE
            ValueError: if emb_activation is not tanh or linear
        """
        super().__init__(mode, iterator, hparams)
        self.input_dropout = hparams.input_dropout
        self.emb_noise = hparams.emb_noise
        
        ### TORCH LAYERS
        # dropout
        # GRU
        self.encoder_dropout = torch.nn.Dropout(p = 1.0 - self.input_dropout, inplace = )   ## p is probability of element to be ZEROED
        self.encoder_GRU = torch.nn.GRU(input_size = , hidden_size = , num_layers =)
        self.dense_layer - nn.Linear(<in>,self.embedding_size,...)

    def _encoder(self, encoder_emb_inp):
        """Method that defines the encoder part of the translation model graph."""
        if self.mode == "TRAIN":
            ### TF SHAPE
            #max_time = tf.shape(encoder_emb_inp)[1]
            max_time = torch.size(encoder_emb_inp)[1]
            ### TF NN DROPOUT
            #encoder_emb_inp = tf.nn.dropout(encoder_emb_inp,
            #                                1. - self.input_dropout,  ## rate, proability elements are set to 0
            #                                noise_shape=[self.batch_size, max_time, 1])
            !!!! no noise_shape parameter in pytorch dropout...
            !!!! This will mean depth will be randomised...
            !!!! Solution -> Implement custom layer?
            encoder_emb_inp = self.encoder_dropout(encoder_emb_inp)

        #encoder_cell = [tf.nn.rnn_cell.GRUCell(size) for size in self.cell_size]
        #encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell)
        #encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
        #                                                   encoder_emb_inp,
        #                                                   sequence_length=self.input_len,
        #                                                   dtype=tf.float32,
        #                                                   time_major=False) ### implies size [batch_size, max_time, depth]
        
        ### EXAMPLE
        #>>> rnn = nn.GRU(10, 20, 2)
        #>>> input = torch.randn(5, 3, 10)
        #>>> h0 = torch.randn(2, 3, 20)
        #>>> output, hn = rnn(input, h0)

        h0 = torch.???() ## probably zero, or random?
        encoder_outputs, encoder_state = self.encoder_GRU(encoder_emb_inp, h0)

        ### TF LAYERS DENSE
        #emb = tf.layers.dense(tf.concat(encoder_state, axis=1),
        #                      self.embedding_size
        #                     )
        emb = self.dense_layer(torch.cat(encoder_state, dim = 1))

        if (self.emb_noise >= 0) & (self.mode == "TRAIN"):
            #emb += tf.random_normal(shape=tf.shape(emb),
            #                        mean=0.0,
            #                        stddev=self.emb_noise,
            #                        dtype=tf.float32)
            emb += torch.normal(mean = torch.zeros(torch.size(emb)), std = self.emb_noise)
        emb = self.emb_activation(emb)
        return emb

class ModelWithGrads(NoisyGRUSeq2SeqWithFeatures):
    def __init__(self, mode, iterator, hparams):
        super().__init__(mode, iterator, hparams)
        
        ### TORCH LAYERS

    def build_graph(self):
        #with tf.name_scope("Input"):
        self.input_seq = tf.placeholder(tf.int32, [None, None])
        self.input_len = tf.placeholder(tf.int32, [None])
        self.start_grads = tf.placeholder(tf.float32, [None, ndims])
        encoder_emb_inp = self._emb_lookup(self.input_seq)

        #with tf.variable_scope("Encoder"):
        self.encoded_seq = self._encoder(encoder_emb_inp)
        ### TF GRADIENTS!
        self.grads = tf.gradients(self.encoded_seq, encoder_emb_inp, self.start_grads)
        #self.grads = torch.gradient(self)
        self.grads = torch.autograd.grad(<Y>,<X>, torch.ones_like(<Y>)) ## Check this 
        ##https://discuss.pytorch.org/t/newbie-pytorch-equivalent-of-tf-tf-gradients-out-in/130988/2
        ## versus
        ## https://www.tensorflow.org/api_docs/python/tf/gradients

        ### TF TRAIN SAVER
        self.saver_op = tf.train.Saver()
        self.saver_op = torch.save() ### Needs more thought
