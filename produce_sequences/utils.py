#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 12:38:10 2018

@author: moffatl, Joe G Greener

utility functions for going to and from matrix and sequence representations

"""

import numpy as np

seq_len = 140
gap_char = "-"
spe_char = "X"

AAS = list("GALMFWKQESPVICYHRNDT")

seq_choices = AAS + [spe_char, gap_char]
seq_index_map = {aa: i for i, aa in enumerate(seq_choices)}
n_symbols = len(seq_choices)


def seq_to_vec(seq):
   assert len(seq) <= seq_len
   seq_ind = [seq_index_map[gap_char]] * seq_len
   for i, aa in enumerate(seq):
       seq_ind[i] = seq_index_map[aa]
   vec = np.zeros(seq_len * n_symbols, dtype=np.uint8)
   for i, j in enumerate(seq_ind):
       vec[i * n_symbols + j] = 1
   return vec


def vec_to_seq(vec):
    "Convert output vector back to human-readable form."
    return "".join(
       seq_choices[vec[i: i + n_symbols].argmax()]
       for i in range(0, seq_len * n_symbols, n_symbols)
    ).replace("-", "")
