#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:32:34 2018

@author: moffatl, Joe G Greener

This script takes in a supplied grammar string e.g. "+B+0-C+0+B+2-B+1" in a 
text document. This then takes this sequence and runs it through a forward pass
of the decoder (the generator).

The only two arguments that need to be passed are the text_file and number of 
sequences out wanted. These are written to standard out

This assumes you are running things on a cpu by default. 
"""
import argparse

import numpy as np
import torch
from lark import Lark

import utils
import vae

# Definition of the grammar
fold_parser = Lark(
    r"""
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
    """,
    start='topology',
)

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


def get_rules(parse_tree):
    "Get the rules that generate a parse tree."
    if parse_tree.data == 'topology':
        if len(parse_tree.children) == 2:
            yield 'T -> E T'
        else:
            yield 'T -> E'
        for child in parse_tree.children:
            yield from get_rules(child)
    elif parse_tree.data == 'element':
        yield 'E -> O L P'
        inner = [child.value for child in parse_tree.children]
        yield 'O -> ' + inner[0]
        yield 'L -> ' + inner[1]
        yield 'P -> ' + inner[2]


def get_vector(topology_string):
    "Get the one-hot encoding vector from a topology string."
    t = fold_parser.parse(topology_string)
    rules = list(get_rules(t))
    v = [0] * max_rules * n_rules
    for i, r in enumerate(rules):
        v[i * n_rules + rules_list.index(r)] = 1
    for i in range(len(rules), max_rules):
        v[i * n_rules + rules_list.index('-')] = 1
    return np.array(v)


def rawGen(model, gram):
    """Generate a raw sequence from the model using ancestral sampling."""
    model.eval()
    # create a empty metal code for the sequence.
    code = torch.zeros(model.batch_size, 8)
    struc = torch.FloatTensor(gram).tile(model.batch_size, 1)
    with torch.no_grad():
        for seq in model.generator(code, struc):
            print(utils.vec_to_seq(seq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-infile", type=str,
            help="file with sequence", default="examples/gram2seq_example.txt")# its either struc or nostruc
    parser.add_argument("-numout", type=int, help="number of sequences generated", default=10)
    args = parser.parse_args()

    model = vae.VariationalAutoEncoder(
        input_size=4353,
        hidden_sizes=[512, 256, 128, 16],
        condition_size=1273,
        batch_size=args.numout,
    )
    state_dict = torch.load("models/grammar16_cutoff.p", map_location=lambda storage, _: storage)
    state_dict = {k.lower(): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # read in the sequence
    with open(args.infile, "r") as f:
        seq = f.readlines()[0].rstrip()

    grammar = get_vector(seq)
    rawGen(model, grammar)


if __name__ == "__main__":
    main()
