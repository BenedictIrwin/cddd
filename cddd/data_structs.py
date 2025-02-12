#!/usr/bin/env python3

import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset
import pandas as pd

from .utils import Variable #, get_moments
#from discriminator.utils import get_headings


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_dict, max_length=140):
        #self.special_tokens = ['</s>', '<s>']
        #self.additional_chars = set()
        #self.chars = self.special_tokens
        #self.vocab_size = len(self.chars)
        #self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab = init_dict
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        self.chars = list(self.vocab.keys())
        #if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['</s>']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        #smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        #regex = '(\[[^\[\]]{1,6}\])'
        REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'

        #smiles = replace_halogen(smiles)
        #print(smiles)
        char_list = re.findall(REGEX_SML, smiles)
        tokenized = ['<s>'] + char_list
        #for char in char_list:
        #    if char.startswith('['):
        #        tokenized.append(char)
        #    else:
        #        chars = [unit for unit in char]
        #        [tokenized.append(unit) for unit in chars]
        tokenized.append('</s>')
        return tokenized

    #def add_characters(self, chars):
    #    """Adds characters to the vocabulary"""
    #    for char in chars:
    #        self.additional_chars.add(char)
    #    char_list = list(self.additional_chars)
    #    char_list.sort()
    #    self.chars = char_list + self.special_tokens
    #    self.vocab_size = len(self.chars)
    #    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    #    self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    #def init_from_file(self, file):
    #    """Takes a file containing \n separated characters to initialize the vocabulary"""
    #    with open(file, 'r') as f:
    #        chars = f.read().split()
    #    self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)


class DatasetWithFeatures():
    #def __init__(self, voc, smi_file, vec_file):
    def __init__(self, model, smi_file_or_list):
        """
        #Args:
        """
        self.voc = model.vocab
        self.hparams = model.hparams
        #self.batch_size = 10 ### For testing purposes
        self.max_length_smiles = 150
        print("REMOVE CONCEPT OF MAX LENGTH SMILES")
        if(isinstance(smi_file_or_list, str)):
          with open(smi_file_or_list) as f: self.smiles = [i.strip() for i in f.readlines()]
        elif(isinstance(smi_file_or_list, list)):
          self.smiles = smi_file_or_list
        else:
          print("DatasetWithFeatures: Argument is list of smiles, or name of file containing smiles")
          exit()
        #self.smiles = pd.read_csv(smi_file, header=0, dtype=str).values
        #self.smiles = [x[0] for x in self.smiles]

        # reading descriptors
        #data = pd.read_csv(vec_file, header=0)
        #look for missing data entries
        #if data.isnull().values.any():
        #    raise ValueError('Found nan in data, possible data missing in generation vectors.')

        #correct heading order
        #headings = get_headings()
        #data = data.reindex(columns=headings)
        #look for missing columns
        #if data.isnull().values.any():
        #    raise ValueError('Found nan in data, possible columns missing in generation vectors.')

        #calculate mew and std for normilization of generation vectors
        #self.vectors = data.values
        #self.mew, self.std = get_moments(self.vectors)
        # catch any zeros which will give nan when normalizing
        #self.std = np.array([x if x != 0 else 1.0 for x in self.std])
        #self.vectors = (self.vectors - self.mew) / self.std

    def __len__(self):
      """ The number of mini-batches in the dataset """
      return int(np.ceil(len(self.smiles)/self.hparams.batch_size))

    def __getitem__(self, i):
      """ Returns mini-batch i as a pair of input tensors """
      #if(self.shuffle):
      #  mols = np.random.choice(self.smiles, size = min(self.hparams.batch_size,len(self.smiles)), replace = False)
      #else:
      #  mols = 
      #vec = self.vectors[i]
      if(i == len(self) - 1):
        """ The final, potentially incomplete mini-batch """
        mols = self.smiles[i*self.hparams.batch_size:]
      else:
        mols = self.smiles[i*self.hparams.batch_size:(i+1)*self.hparams.batch_size]

      tokenized = [self.voc.tokenize(mol) for mol in mols]
      encoded = np.array([self.voc.encode(t) for t in tokenized])
      lengths = np.array([len(enc) for enc in encoded])
      encoded = np.array([np.pad(enc,(0,self.hparams.max_length_smiles - len(enc))) for enc in encoded])
      #return [Variable(encoded), Variable(vec)]
      return Variable(encoded).int(), Variable(lengths).int()

    #@classmethod
    #def collate_fn(cls, arr):
    #    """Function to take a list of encoded sequences and turn them into a batch"""
    #    max_length_smi = max([data[0].size(0) for data in arr])
    #    #max_length_vec = max([data[1].size(0) for data in arr])
    #    collated_arr_smi = Variable(torch.zeros(len(arr), max_length_smi))
    #    #collated_arr_vec = Variable(torch.zeros(len(arr), max_length_vec))
    #    for i, data in enumerate(arr):
    #        collated_arr_smi[i, :data[0].size(0)] = data[0]
    #        #collated_arr_vec[i, :data[1].size(0)] = data[1]
    #    return collated_arr_smi#, collated_arr_vec

#class Experience(object):
#    """Class for prioritized experience replay that remembers the highest scored sequences
#       seen and samples from them with probabilities relative to their scores."""
#    def __init__(self, voc, max_size=100):
#        self.memory = []
#        self.max_size = max_size
#        self.voc = voc
#
#    def add_experience(self, experience):
#        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
#        self.memory.extend(experience)
#        if len(self.memory)>self.max_size:
#            # Remove duplicates
#            idxs, smiles = [], []
#            for i, exp in enumerate(self.memory):
#                if exp[0] not in smiles:
#                    idxs.append(i)
#                    smiles.append(exp[0])
#            self.memory = [self.memory[idx] for idx in idxs]
#            # Retain highest scores
#            self.memory.sort(key = lambda x: x[1], reverse=True)
#            self.memory = self.memory[:self.max_size]
#            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))
#
#    def sample(self, n):
#        """Sample a batch size n of experience"""
#        if len(self.memory)<n:
#            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
#        else:
#            scores = [x[1] for x in self.memory]
#            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
#            sample = [self.memory[i] for i in sample]
#            smiles = [x[0] for x in sample]
#            scores = [x[1] for x in sample]
#            prior_likelihood = [x[2] for x in sample]
#        tokenized = [self.voc.tokenize(smile) for smile in smiles]
#        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
#        encoded = MolData.collate_fn(encoded)
#        return encoded, np.array(scores), np.array(prior_likelihood)
#
#    def initiate_from_file(self, fname, scoring_function, Prior):
#        """Adds experience from a file with SMILES
#           Needs a scoring function and an RNN to score the sequences.
#           Using this feature means that the learning can be very biased
#           and is typically advised against."""
#        with open(fname, 'r') as f:
#            smiles = []
#            for line in f:
#                smile = line.split()[0]
#                if Chem.MolFromSmiles(smile):
#                    smiles.append(smile)
#        scores = scoring_function(smiles)
#        tokenized = [self.voc.tokenize(smile) for smile in smiles]
#        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
#        encoded = MolData.collate_fn(encoded)
#        prior_likelihood, _ = Prior.likelihood(encoded.long())
#        prior_likelihood = prior_likelihood.data.cpu().numpy()
#        new_experience = zip(smiles, scores, prior_likelihood)
#        self.add_experience(new_experience)
#
#    def print_memory(self, path):
#        """Prints the memory."""
#        print("\n" + "*" * 80 + "\n")
#        print("         Best recorded SMILES: \n")
#        print("Score     Prior log P     SMILES\n")
#        with open(path, 'w') as f:
#            f.write("SMILES Score PriorLogP\n")
#            for i, exp in enumerate(self.memory[:100]):
#                if i < 50:
#                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
#                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
#        print("\n" + "*" * 80 + "\n")
#
#    def __len__(self):
#        return len(self.memory)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print('Found invalid smiles in training set, i=={}, removing from set'.format(i))
            else:
                #Removed filter, we are interested in everything
                smiles_list.append(Chem.MolToSmiles(mol))   
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list

def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6,7,8,9,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    return smiles_list_valid

def filter_file_on_chars(smiles_fname, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(smiles_fname + "_filtered", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")

def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open('data/Voc', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_file(smiles_file)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")
