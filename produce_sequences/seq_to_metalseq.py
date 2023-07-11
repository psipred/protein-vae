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
from sklearn.metrics import accuracy_score

import utils
import vae


def newMetalBinder(model, data, code):
    """
    Generate a new sequence based on a metal code.

    The first 3080 dims are the sequence, the final 8 are the metal binding
    flags. Fold is optional.
    """
    model.eval()
    
    batch_size = model.batch_size
    code = torch.FloatTensor(code).tile(batch_size, 1)
    x = torch.FloatTensor(data[:3080]).tile(batch_size, 1)
    with torch.no_grad():
        x_sample, *_ = model(x, code)

    y_label = x.reshape(batch_size, -1, 22).argmax(-1)
    y_pred = x_sample.reshape(batch_size, -1, 22).argmax(-1)

    scores = [
        accuracy_score(row[:row.argmax()], y_pred[i][:row.argmax()])
        for i, row in enumerate(y_label)
    ]
    print(f"Average Sequence Identity to Input: {np.mean(scores):.1%}")
    
    out_seqs = x_sample.cpu().numpy()
    for seq in out_seqs:
        print(utils.vec_to_seq(seq))


def load_model(path="models/metal16_nostruc", batch_size=1):
    model = vae.VariationalAutoEncoder(
        input_size=3088,
        hidden_sizes=[512, 256, 128, 16],
        condition_size=8,
        batch_size=batch_size,
    )
    state_dict = torch.load(path, map_location=lambda storage, _: storage)
    state_dict = {k.lower(): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


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

    model = load_model(batch_size=args.numout)
    code = np.zeros(8, dtype=np.uint8)
    code[metals_dict[args.metal]] = 1
    for vec in parse_fasta(args.infile):
        print(f"Input sequence:\n{utils.vec_to_seq(vec)}\n")
        newMetalBinder(model, vec, code)
        print()


if __name__ == "__main__":
    main()
