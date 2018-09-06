#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:32:34 2018

@author: moffatl

This script takes in a supplied grammar string e.g. "+B+0-C+0+B+2-B+1" in a 
text document. This then takes this sequence and runs it through a forward pass
of the decoder (the generator).

The only two arguments that need to be passed are the text_file and number of 
sequences out wanted. These are written to standard out

This assumes you are running things on a cpu by default. 

"""



# =============================================================================
# Imports
# =============================================================================

import torch
import torch.nn.functional as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import argparse
import utils
from lark import Lark

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

# =============================================================================
# Sort out Command Line arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-infile", type=str,
        help="file with sequence", default="examples/gram2seq_example.txt")# its either struc or nostruc
parser.add_argument("-numout", type=int,
        help="number of sequences generated", default=10)

args = parser.parse_args()
args_dict = vars(args)        


# =============================================================================
# Pytorch Module
# =============================================================================
class VAE(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, batch_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        

        self.fc = torch.nn.Linear(input_size, hidden_sizes[0])  
        self.BN = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.BN1 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.BN2 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        
        self.fc4 = torch.nn.Linear(hidden_sizes[3]+1273, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size-1273)
    
    
    def sample_z(self,x_shape, mu, log_var):
        # Using reparameterization trick to sample from a gaussian    
        eps = Variable(torch.randn(x_shape, self.hidden_sizes[-1]))
        return mu + torch.exp(log_var / 2) * eps    
    
    def forward(self, x, code, struc):
        
        ###########
        # Encoder #
        ###########
        # get the code from the tensor
        # add the conditioned code
        
        x = torch.cat((x,code,struc),1)        
        # Layer 0
        out1 = self.fc(x)
        out1 = nn.relu(self.BN(out1))
        # Layer 1
        out2 = self.fc1(out1)
        out2 = nn.relu(self.BN1(out2))
        # Layer 2
        out3 = self.fc2(out2)
        out3 = nn.relu(self.BN2(out3))
        # Layer 3 - mu
        mu   = self.fc3_mu(out3)
        # layer 3 - sig
        sig  = nn.softplus(self.fc3_sig(out3))        


        ###########
        # Decoder #
        ###########
        
        # sample from the distro
        sample= self.sample_z(x.size(0),mu, sig)
        # add the conditioned code
        sample = torch.cat((sample, code, struc),1)
        # Layer 4
        out4 = self.fc4(sample)
        out4 = nn.relu(self.BN4(out4))
        # Layer 5
        out5 = self.fc5(out4)
        out5 = nn.relu(self.BN5(out5))
        # Layer 6
        out6 = self.fc6(out5)
        out6 = nn.relu(self.BN6(out6))
        # Layer 7
        out7 = nn.sigmoid(self.fc7(out6))
        
        return out7, mu, sig
    
    
    
    
    def generator(self,num_seq,code,s):
        
        ###########
        # Decoder #
        ###########

        z = Variable(torch.randn(num_seq, self.hidden_sizes[-1]))
        # add the conditioned code
        sample = torch.cat((z, code, s),1)
        # Layer 4
        out4 = self.fc4(sample)
        out4 = nn.relu(self.BN4(out4))
        # Layer 5
        out5 = self.fc5(out4)
        out5 = nn.relu(self.BN5(out5))
        # Layer 6
        out6 = self.fc6(out5)
        out6 = nn.relu(self.BN6(out6))
        # Layer 7
        out7 = nn.sigmoid(self.fc7(out6))
        
        return out7
    
    def encoder(self, x, code, s):
        
        x = torch.cat((x,code,s),1)        
        # Layer 0
        out1 = self.fc(x)
        out1 = nn.relu(self.BN(out1))
        # Layer 1
        out2 = self.fc1(out1)
        out2 = nn.relu(self.BN1(out2))
        # Layer 2
        out3 = self.fc2(out2)
        out3 = nn.relu(self.BN2(out3))
        # Layer 3 - mu
        mu   = self.fc3_mu(out3)
        
        return mu

# =============================================================================
# Create and Load model into memory
# =============================================================================

X_dim=4353
hidden_size=[512,256,128,16]
batch_size=args_dict["numout"]
vae = VAE(X_dim, hidden_size, batch_size)
# load model
vae.load_state_dict(torch.load("models/grammar16_cutoff", map_location=lambda storage, loc: storage))

# =============================================================================
#  Define function to produce sequences. 
# =============================================================================

def rawGen(model, gram):
    """
    Generates a raw sequence from the model using ancestral sampling
    """
    
    #model into eval mode    
    model.eval()
    C=np.zeros((batch_size,8))
    S=np.tile(gram,(batch_size,1))
    #create a empty metal code for the sequence.
    C=torch.from_numpy(C).type(torch.FloatTensor)
    S=torch.from_numpy(S).type(torch.FloatTensor)

    x_sample=model.generator(batch_size,C,S)    
    len_aa=140*22
    out_seqs=x_sample[:,:len_aa].cpu().data.numpy()
    for seq in out_seqs:
        print(utils.vec_to_seq(seq))
        
    return




# =============================================================================
# Produce new sequence
# =============================================================================

# first we read in the sequence from the 
with open(args_dict["infile"],'r') as in_file:
    seq=in_file.readlines()

seq=seq[0]

# Definition of the grammar
fold_parser = Lark(r"""
    topology: element topology
            | element

    element: ORIENTATION LAYER POSITION

    ORIENTATION: /\+/
               | /-/

    LAYER: /A/
         | /B/
         | /C/
         | /E/

    POSITION: /\+[0-6]/
            | /-[1-6]/

    %ignore /\.+/

    """, start='topology')

# List of grammar production rules
rules_list = [
    'T -> E T',
    'T -> E',
    'E -> O L P',
    'O -> +',
    'O -> -',
    'L -> A',
    'L -> B',
    'L -> C',
    'L -> E',
    'P -> +0',
    'P -> +1',
    'P -> +2',
    'P -> +3',
    'P -> +4',
    'P -> +5',
    'P -> +6',
    'P -> -1',
    'P -> -2',
    'P -> -3',
    'P -> -4',
    'P -> -5',
    'P -> -6',
    '-', # Blank for no rule
]

n_rules = len(rules_list)
max_rules = 55 # 5 rules per element, the longest string has 11 elements

# Indices of valid rules starting from each symbol
valid_rules = {
    'T': [0, 1],
    'E': [2],
    'O': [3, 4],
    'L': [5, 6, 7, 8],
    'P': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}

# Get the rules that generate a parse tree
def get_rules(t):
    if t.data == 'topology':
        c = t.children
        if len(c) == 2:
            yield 'T -> E T'
        else:
            yield 'T -> E'
        for i in c:
            yield from get_rules(i)
    elif t.data == 'element':
        yield 'E -> O L P'
        inner = [i.value for i in t.children]
        yield 'O -> ' + inner[0]
        yield 'L -> ' + inner[1]
        yield 'P -> ' + inner[2]

# Get the one-hot encoding vector from a topology string
def get_vector(top_string):
    t = fold_parser.parse(top_string)
    rules = list(get_rules(t))
    v = [0] * max_rules * n_rules
    for i, r in enumerate(rules):
        v[i * n_rules + rules_list.index(r)] = 1
    for i in range(len(rules), max_rules):
        v[i * n_rules + rules_list.index('-')] = 1
    return np.array(v)

grammar = get_vector(seq)

# =============================================================================
# Produce new sequence
# =============================================================================

rawGen(vae, grammar)

    
    