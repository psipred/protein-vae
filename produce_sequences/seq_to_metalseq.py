#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:32:14 2018


@author: Lewis Iain Moffat


This script takes in a supplied sequence (as yet does not do multiple sequences
) that is in a fasta file or a text document. This then takes this sequence and 
runs it through a forward pass of the autoencoder, encoding it and decoding it. 
This produces the same sequence with variation added. 

The three arguments that need to be passed are the text_file and number of 
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

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

# =============================================================================
# Sort out Command Line arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-infile", type=str,
        help="file with sequence", default="examples/seq2metalseq_example.txt")# its either struc or nostruc
parser.add_argument("-numout", type=int,
        help="number of sequences generated", default=10)
parser.add_argument("-metal", type=str,
        help="one of: Fe, Zn, Ca, Na, Cu, Mg, Cd, Ni", default="Fe")

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
           

        self.fc = torch.nn.Linear(input_size, hidden_sizes[0])  # 2 for bidirection 
        self.BN = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.BN1 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.BN2 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        
        self.fc4 = torch.nn.Linear(hidden_sizes[3]+8, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size-8)

    def sample_z(self,x_size, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(x_size, self.hidden_sizes[-1])	
        return mu + torch.exp(log_var / 2) * eps
    
    def forward(self, x, code, struc=None):
        
        ###########
        # Encoder #
        ###########
        
        # get the code from the tensor
        # add the conditioned code
        x = torch.cat((x,code),1)    
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
        sample = torch.cat((sample, code),1)
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



# =============================================================================
# Create and Load model into memory
# =============================================================================

X_dim=3088
hidden_size=[512,256,128,16]
batch_size=args_dict["numout"]
vae = VAE(X_dim, hidden_size, batch_size)
# load model
vae.load_state_dict(torch.load("models/metal16_nostruc", map_location=lambda storage, loc: storage))

# =============================================================================
#  Define function to produce sequences. 
# =============================================================================

    
def newMetalBinder(code,model,data):
    """
    Generates a new sequence based on a metal code; the first 3080 dims are the
    sequence, the final 8 are the metal binding flags. Fold is optional
    """
    scores=[]
    model.eval()
    
    code = np.tile(code,(model.batch_size,1))
    x = np.tile(data[:3080],(model.batch_size,1))
    X = torch.from_numpy(x).type(torch.FloatTensor)
    C = torch.from_numpy(code).type(torch.FloatTensor)

    x_sample, z_mu, z_var = model(X, C)
    
    
    len_aa=140*22
    y_label=np.argmax(x[:,:len_aa].reshape(batch_size,-1,22), axis=2)
    y_pred=np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    for idx, row in enumerate(y_label):
        scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Average Sequence Identity to Input: {0:.1f}%".format(np.mean(scores)*100))
    
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

# format the sequence so if it is a FASTA file then we turf the line with >
for idx, line in enumerate(seq):
    seq[idx]=line.replace("\n","")

seq_in=""
for line in seq:
    if ">" in line:
        continue
    else:
        seq_in=seq_in+line

# now have a string which is the sequence        
seq_in_vec=utils.seq_to_vec(seq_in)

# now we want to create the right metal code as supplied
metals=['Fe', 'Zn', 'Ca', 'Na', 'Cu', 'Mg', 'Cd', 'Ni']
metals_dict={}
for idx, metal in enumerate(metals): metals_dict[metal]=idx
try:
    code=np.zeros(8)
    code[metals_dict[args_dict["metal"]]]=1
except:
    print("Please supply one of the correct 8 names for metals")

newMetalBinder(code,vae,seq_in_vec)


    
    
    
    
    
    
    