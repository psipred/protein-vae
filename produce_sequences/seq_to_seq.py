#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:31:34 2018

@author: Lewis Iain Moffat


This script takes in a supplied sequence (as yet does not do multiple sequences
) that is in a fasta file or a text document. This then takes this sequence and 
runs it through a forward pass of the autoencoder, encoding it and decoding it. 
This produces the same sequence with variation added. This presumes the protein
sequence is not a metal binder however if it is it shouldn't affect the 
variation. This is because the metal binding variational model is used instead
of the grammar model. This is for simplicities sake i.e. you don't need to have
the grammar of a protein before being able to run this script. 

The only two arguments that need to be passed are the text_file and number of 
sequences out wanted. These are written to standard out

This assumes you are running things on a cpu by default. 
"""
import argparse

import numpy as np

import utils
from seq_to_metalseq import load_model, newMetalBinder, parse_fasta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str,
            help="file with sequence", default="examples/seq2seq_example.txt")
    parser.add_argument("--numout", type=int,
            help="number of sequences generated", default=10)
    args = parser.parse_args()

    model = load_model(batch_size=args.numout)
    code = np.zeros(8, dtype=np.uint8)
    for vec in parse_fasta(args.infile):
        print(f"Input sequence:\n{utils.vec_to_seq(vec)}\n")
        newMetalBinder(model, vec, code)
        print()


if __name__ == "__main__":
    main()
