#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 12:38:10 2018

@author: moffatl, Joe G Greener

utility functions for going to and from matrix and sequence representations

"""

import numpy as np

seq_len = 140
gap_char = '-'
spe_char = 'X'

aas = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
      'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']

seq_choices = aas + [spe_char, gap_char]
n_symbols = len(seq_choices)

def seq_to_vec(seq):
   assert len(seq) <= seq_len
   seq_ind = [seq_choices.index(gap_char)] * seq_len
   for i, aa in enumerate(seq):
       seq_ind[i] = seq_choices.index(aa)
   vec = [0] * seq_len * n_symbols
   for i, j in enumerate(seq_ind):
       vec[i * n_symbols + j] = 1
   return np.array(vec)


# Convert output vector back to human-readable form
def vec_to_seq(vec):
    seq_info = list(vec[:3080])
    seq = ""
    for i in range(seq_len):
        seq += seq_choices[np.argmax(seq_info[i*n_symbols:(i+1)*n_symbols])]
    
    #seq=seq[:seq.find("-")]
    seq=seq.replace("-","")
    
    return seq
