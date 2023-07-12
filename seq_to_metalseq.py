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
import argparse

import numpy as np
import torch

import utils
import vae


def newMetalBinder(model, data, code, num_samples):
    """
    Generate a new sequence based on a metal code.

    The first 3080 dims are the sequence, the final 8 are the metal binding
    flags. Fold is optional.
    """
    model.eval()
    code = torch.as_tensor(code).float().tile(num_samples, 1)
    x = torch.as_tensor(data[:3080]).float().tile(num_samples, 1)
    with torch.no_grad():
        x_sample, *_ = model(x, code)

    y_label = model.extract_label(x)
    y_pred = model.extract_label(x_sample)
    scores = model.compute_scores(y_pred, y_label)
    print(f"Average Sequence Identity to Input: {np.mean(scores):.1%}")
    
    for seq in x_sample:
        print(utils.vec_to_seq(seq))


def parse_fasta(path):
    # format the sequence so if it is a FASTA file then we turf the line with >
    with open(path, "r") as f:
        for chunk in f.read().split("\n\n"):
            seq = "".join(
                line.strip().rstrip("*")
                for line in chunk.splitlines()
                if line.strip().rstrip("*").isupper()
            )
            if seq:
                yield utils.seq_to_vec(seq)


def main():
    metals = ["Fe", "Zn", "Ca", "Na", "Cu", "Mg", "Cd", "Ni"]
    metals_dict = {metal: i for i, metal in enumerate(metals)}

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str,
            help="file with sequence", default="examples/seq2metalseq_example.txt")
    parser.add_argument("--numout", type=int,
            help="number of sequences generated", default=10)
    parser.add_argument("--metal", type=str, choices=metals, default="Fe")
    args = parser.parse_args()

    model = vae.make_autoencoder(False, 16, "models/metal16_nostruc.p")
    code = np.zeros(8, dtype=np.uint8)
    code[metals_dict[args.metal]] = 1
    for vec in parse_fasta(args.infile):
        print(f"Input sequence:\n{utils.vec_to_seq(vec)}\n")
        newMetalBinder(model, vec, code, args.numout)
        print()


if __name__ == "__main__":
    main()
