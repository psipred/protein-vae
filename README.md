# Variational autoencoder for protein sequences

This repository provides code to accompany the paper:

Design of metalloproteins and novel protein folds using variational autoencoders - link

The work describes a variational autoencoder that can add metal binding sites to protein sequences, or generate protein sequences for a given protein topology.

## Dependencies
* python 3.6
* pytorch 3.1
* sklearn 0.19.1
* numpy 1.14.1

## Usage

The two tasks this work approaches is adding a metal binding site to a protein sequence and generating a protein sequence for a given topology string. The first is described as Task 1 and the second as Task 2. Below are descriptions for using the trained models.

### Task 1
This is adding a metal binding site to a protein sequence. The files for this task as located in the metal_gen folder. Aside from the model file in the folder folder, the main script is the `metal_VAE_pytorch.py`. This file can be run from the command line with several arguments that can be seen by looking at the source. The code itself contains explanations for its use, but more specifically, it can be used for training a model or producing samples of a protein similar to another protein provided. 

<b> Training </b>

In the case of training please get in contact for the datasets, however if you have correctly formated data you should be able to use that instead.

For inference we have provided 

To encode/decode a protein sequence:

To add a metal binding site to a protein sequence:

To generate a protein sequence for a given topology string:

