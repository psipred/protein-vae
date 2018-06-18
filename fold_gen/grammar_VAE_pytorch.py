#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:53:06 2017

@author: Lewis Moffat aka. Groovy Dragon


This script is an implimentation of a variational autoencoder in pytorch
This isnt actually hierarchical it just uses a bunch of layers. Gave up on 
bothering with more latent variables

CURRENT STATUS: WORKING

"""


# =============================================================================
# Imports
# =============================================================================

import torch
import torch.nn.functional as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
#import dummyDataGen as dataGen
#import visualizer as vs

from sklearn.metrics import accuracy_score
import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts
# =============================================================================
# Initial Specs and Data Loading
# =============================================================================



#switches -- cuda is if you are running on gpu
want_data=False
cuda=False
load=True
train=False
test=False
sanity_check=True
generateNew=False
encode_it=False
new_metal=False



if want_data:
    if cuda:
        data = np.load('/scratch0/DeNovo/assembled_data_fold_cutoff.npy')        
    else: 
        data = np.load('assembled_data_fold_cutoff.npy')
    
    # sanity check
    print(data.shape)
    
    #for later looping
    n=data.shape[0]

    #data dimension
X_dim = 5053#data.shape[1]
y_dim = 5053#data.shape[1]


if cuda:
    torch.cuda.set_device(2)



#spec batch size
batch_size=1000


"""
[batch x 140 x 22]
[batch x 3080]
fc[batch x 1024]
fc[batch x 512]
fc[batch x 256]
z --> [batch x 128]
fc[batch x 256]
fc[batch x 512]
fc[batch x 1024]
fc[batch x 3080]
[batch x 140 x 22]


"""

#learning rate
lr=5e-4
# layer sizes
hidden_size=[512,256,128,16]



# =============================================================================
# Module
# =============================================================================
class feed_forward(torch.nn.Module):
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

    
    def conv_size(self,W,F,P=1,S=1):
        """
        Function to calculate new size of 2nd dim in 1D convolution layers
        """
        return ((W-F+(2*P))/S)+1
    
    
    
    
    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        
        if cuda:
            eps = Variable(torch.randn(self.batch_size, self.hidden_sizes[-1])).cuda()
        else:
            eps = Variable(torch.randn(self.batch_size, self.hidden_sizes[-1]))
	
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
        sample= self.sample_z(mu, sig)
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
#        # Layer 7
#        out7=self.fc7(out6)
#        out7 = out7.view(-1,22,140)
#        out7 = softmax(out7,axis=1)
        # Layer 7
        out7 = nn.sigmoid(self.fc7(out6))
        
        return out7, mu, sig
    
    
    
    
    def generator(self,code,s):
        
        ###########
        # Decoder #
        ###########

        z = Variable(torch.randn(self.batch_size, self.hidden_sizes[-1]))
        if cuda:
            z.cuda()
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
   
    
    
    
def softmax(input, axis=1):
    input_size = input.size()
    
    trans_input = input.transpose(axis, len(input_size)-1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    
    soft_max_2d = F.softmax(input_2d)
    
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size)-1)    

# =============================================================================
# Training
# =============================================================================


# the -700 is a hacky way to deal with dropping the secondary structure
if cuda:
    ff = feed_forward(X_dim-700, hidden_size, batch_size).cuda()
else:
    ff = feed_forward(X_dim-700, hidden_size, batch_size)

if load:
    ff.load_state_dict(torch.load("models/grammar16_cutoff", map_location=lambda storage, loc: storage))


# Loss and Optimizer
solver = optim.Adam(ff.parameters(), lr=lr)
burn_in_counter =0
tick = 0
epoch_prog=[]

# number of epochs
num_epochs=1000

if train:
#    for its in range(num_epochs):
#        ff.train()
#        scores=[]
#        data=shuffle(data)
#        print("Grammar Cond. - Epoch: {0}/{1}  Latent: {2}".format(its,num_epochs,hidden_size[-1]))
#        for it in range(n // batch_size):
#            
#            x_batch=data[it * batch_size: (it + 1) * batch_size]
#            code = x_batch[:,-8:]
#            structure = x_batch[:,3780:5045]
#            x_batch = x_batch[:,:3080]
#    
#            if cuda:
#                X = Variable(torch.from_numpy(x_batch)).cuda().type(torch.cuda.FloatTensor)
#                C = Variable(torch.from_numpy(code)).cuda().type(torch.cuda.FloatTensor)
#                S = Variable(torch.from_numpy(structure)).cuda().type(torch.cuda.FloatTensor) 
#            else:
#                X = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor)
#                C = Variable(torch.from_numpy(code)).type(torch.FloatTensor)
#                S = Variable(torch.from_numpy(structure)).type(torch.FloatTensor)
#            #turf last gradients
#            solver.zero_grad()
#            
#            # Forward
#            x_sample, z_mu, z_var = ff(X, C, S)
#            
#            
#            # Loss
#            recon_loss = nn.binary_cross_entropy(x_sample, X, size_average=False) # by setting to false it sums instead of avg.
#            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
#            kl_loss = kl_loss*burn_in_counter
#            loss = recon_loss + kl_loss
#            
#            
#            # Backward
#            loss.backward()
#        
#            # Update
#            solver.step()
#            
#            
#            
#            len_aa=140*22
#            y_label=np.argmax(x_batch[:,:len_aa].reshape(batch_size,-1,22), axis=2)
#            y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
#            
#            
#            # can use argmax again for clipping as it uses the first instance of 21
#            # loop with 256 examples is only about 3 milliseconds                      
#            for idx, row in enumerate(y_label):
#                scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
#        
#        print("Tra Acc: {0}".format(np.mean(scores)))
#        epoch_prog.append(np.mean(scores))        
#        
#        if its>300 and burn_in_counter<1.0:
#            burn_in_counter+=0.003
#
        scores=[]
        
        ff.eval()
        for it in range(data.shape[0] // batch_size):
            x_batch=data[it * batch_size: (it + 1) * batch_size]
            code = x_batch[:,-8:]
            structure = x_batch[:,3780:5045]
            x_batch = x_batch[:,:3080]



            if cuda:
                X = Variable(torch.from_numpy(x_batch)).cuda().type(torch.cuda.FloatTensor)
                C = Variable(torch.from_numpy(code)).cuda().type(torch.cuda.FloatTensor)
                S = Variable(torch.from_numpy(structure)).cuda().type(torch.cuda.FloatTensor)
            else:
                X = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor)
                C = Variable(torch.from_numpy(code)).type(torch.FloatTensor)
                S = Variable(torch.from_numpy(structure)).type(torch.FloatTensor)
                
            # Forward
            x_sample, z_mu, z_var = ff(X, C, S)

        
            len_aa=140*22
            y_label=np.argmax(x_batch[:,:len_aa].reshape(batch_size,-1,22), axis=2)
            y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)

            for idx, row in enumerate(y_label):
                scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
        print("Val Acc: {0}".format(np.mean(scores)))






def encoderQuick(model, data, name):
    model.eval()
    newRep=np.zeros((batch_size,16))
    for it in range(n // batch_size):
        x_batch=data[it * batch_size: (it + 1) * batch_size]
        code = x_batch[:,-8:]
        structure = x_batch[:,3780:5045]
        x_batch = x_batch[:,:3080]
        
        X = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor)
        C = Variable(torch.from_numpy(code)).type(torch.FloatTensor)
        S = Variable(torch.from_numpy(structure)).type(torch.FloatTensor)
        newRep = np.concatenate((newRep,model.encoder(X,C,S).data.numpy()),0)
    newRep=newRep.reshape((-1,16))
    #np.savetxt("reps.tsv",newRep,delimiter='\t')
    return newRep




def sanity(model, test_point,save_name,num_samp=50 ):
    """
    takes the model and a data point and run its through the system to get a number of new samples
    """
    #model into eval mode    
    model.eval()
    
    # rejig it so we have 50 examples of the protein 
    test_point=np.tile(test_point,(num_samp,1))
    code_t = np.zeros((test_point.shape[0],8))
    structure_t = test_point[:,3080:]
    test_point = test_point[:,:3080]
    

    
    X = Variable(torch.from_numpy(test_point)).type(torch.FloatTensor)
    C = Variable(torch.from_numpy(code_t)).type(torch.FloatTensor)
    S = Variable(torch.from_numpy(structure_t)).type(torch.FloatTensor)
                
    # Forward
    x_sample, z_mu, z_var = model(X, C, S)

    len_aa=140*22
    
    y_label=np.argmax(test_point[:,:len_aa].reshape(batch_size,-1,22), axis=2)
    y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    scores=[]
    
    for idx, row in enumerate(y_label):
        scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Val Acc: {0}".format(np.mean(scores)))
    
    np.save(save_name+"_samples",y_pred)
    return 


def newMetalBinder(model,data,name):
    """
    Generates a new sequence based on a metal code and a grammar. 
    The data array is (4353,) where the first 3080 are the
    sequence, the next 1265 are the fold and the final 8 are the metal
    binding flags.
    """
    scores=[]
    #model into eval mode    
    model.eval()
    # split the data    
    x=np.tile(data[:3080],(model.batch_size,1))
    C=np.tile(data[-8:],(model.batch_size,1))
    S=np.tile(data[3080:-8],(model.batch_size,1))
    # create tensors of the data
    C=Variable(torch.from_numpy(C)).type(torch.FloatTensor)
    S=Variable(torch.from_numpy(S)).type(torch.FloatTensor)
    X=Variable(torch.from_numpy(x)).type(torch.FloatTensor)
    
    # Forward
    x_sample, _, _ = model(X, C, S)
    
    len_aa=140*22
    y_label=np.argmax(x[:,:len_aa].reshape(batch_size,-1,22), axis=2)
    y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    np.save(name,y_pred)
    
    for idx, row in enumerate(y_label):
        scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Accuracy: {0}".format(np.mean(scores)))
    
    return



def rawGen(model, gram, name):
    """
    Generates a raw sequence from the model using ancestral sampling
    """
    
    #model into eval mode    
    model.eval()
    C=np.zeros((model.batch_size,8))
    S=np.tile(gram,(model.batch_size,1))
    #create a empty metal code for the sequence.
    C=Variable(torch.from_numpy(C)).type(torch.FloatTensor)
    S=Variable(torch.from_numpy(S)).type(torch.FloatTensor)
    
    x_sample=model.generator(C,S)    
    len_aa=140*22
    y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    np.save(name,y_pred)

    return
    

def accTest(model, data):
    
    scores=[]
        
    model.eval()
    for it in range(data.shape[0] // model.batch_size):
        x_batch=data[it * model.batch_size: (it + 1) * model.batch_size]
        code = x_batch[:,-8:]
        structure = x_batch[:,3780:5045]
        x_batch = x_batch[:,:3080]



        if cuda:
            X = Variable(torch.from_numpy(x_batch)).cuda().type(torch.cuda.FloatTensor)
            C = Variable(torch.from_numpy(code)).cuda().type(torch.cuda.FloatTensor)
            S = Variable(torch.from_numpy(structure)).cuda().type(torch.cuda.FloatTensor)
        else:
            X = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor)
            C = Variable(torch.from_numpy(code)).type(torch.FloatTensor)
            S = Variable(torch.from_numpy(structure)).type(torch.FloatTensor)
            
        # Forward
        x_sample, z_mu, z_var = ff(X, C, S)

    
        len_aa=140*22
        y_label=np.argmax(x_batch[:,:len_aa].reshape(batch_size,-1,22), axis=2)
        y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)

        for idx, row in enumerate(y_label):
            scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Val Acc: {0}".format(np.mean(scores)))
    return scores


if sanity_check:
    ff.eval()
    dp1=np.load("seq_in.npy")
    sanity(ff,dp1,"seq_in_lys" ,batch_size)
    
if generateNew:
    g1=np.load("top_4.npy")
    rawGen(ff,g1,"top_4_samples")

if encode_it:
    reps=encoderQuick(ff,data,"forArtProj")
    
if new_metal:
#
#    g1=np.load("2X1B_in_cu.npy")
#    newMetalBinder(ff,g1,"2X1B_out_cu")

    g1=np.load("prots_nomet.npy")
    for idx, row in enumerate(g1):   
        newMetalBinder(ff,row,"prots_nomet_"+str(idx))

    

# saves if its running on gpu          
if cuda:
    torch.save(ff.state_dict(), 'grammar16')

           
