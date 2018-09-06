#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:53:06 2017

@author: Lewis Moffat aka. Groovy Dragon

This script is an implimentation of a variational autoencoder in pytorch

This script presumes that if you are training the model you are going to 
have a gpu available. If you do not, and you are just running on cpu then it 
assumes it is running in inference. This is very straight forward to change 
should you like. 


The two datasets this can be trained on are assembled_data_mbflip.npy 
and assembled_data_mbflip_fold.npy. They are called nostruc and struc
respectively. The struc set contains 1265 extra dimensions per datum that 
describes the fold grammar. Either can be used but it needs to be specified in
input args. 


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
Variable(
        
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

# =============================================================================
# Sort out Command Line arguments
# =============================================================================


parser = argparse.ArgumentParser()
parser.add_argument("-lr", type=float,
        help="lr", default=5e-4)
parser.add_argument("-batch_size", type=int,
        help="batch_size train", default=10000)
parser.add_argument("-batch_size_test", type=int,
        help="batch_size test", default=25)
parser.add_argument("-num_epochs", type=int,
        help="num_epochs", default=1000)
parser.add_argument("-latent_dim", type=int,
        help="latent_dim", default=16)
parser.add_argument("-device", type=int,
        help="device", default=0)
parser.add_argument("-dataset", type=str,
        help="dataset", default="nostruc")# its either struc or nostruc
args = parser.parse_args()
args_dict = vars(args)        



# =============================================================================
# Switches for what you want the model to do
# =============================================================================
cuda=False       # for training with gpu, make it true. For inference with cpu make false
load=True        # load in the model (default provided is 16 dimensional for nostruc data)
train=False      # Make true to train the model presuming you have the dataset
new_metal=True   # Make true to produce 'batch_size' samples of a given protein
                     # see the docs on github for description of how to do this
                     

# =============================================================================
# Dataset loading and specifying values
# =============================================================================

if cuda:
    if args_dict["dataset"]=="nostruc":    
        data = np.load('/scratch0/DeNovo/assembled_data_mbflip.npy')
    else:
        data = np.load('/scratch0/DeNovo/assembled_data_mbflip_fold.npy')
    print(data.shape)

    data, data_test=tts(data, test_size=0.15, shuffle=True)

    n=data.shape[0]
else:
    print("No DATA")
    if args_dict["dataset"]=="nostruc":
        X_dim=3088
    else:
        X_dim=4353


if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args_dict['device'])

#spec batch size
batch_size=args_dict['batch_size']
#learning rate
lr=args_dict['lr']
# layer sizes
hidden_size=[512,256,128,args_dict['latent_dim']]




# =============================================================================
# Module
# =============================================================================
class feed_forward(torch.nn.Module):
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
        
        if args_dict["dataset"]=="struc":
            self.fc4 = torch.nn.Linear(hidden_sizes[3]+1273, hidden_sizes[2])
        else:        
            self.fc4 = torch.nn.Linear(hidden_sizes[3]+8, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        if args_dict["dataset"]=="struc":
            self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size-1273)
        else:
            self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size-8)

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        
        if cuda:
            eps = torch.randn(self.batch_size, self.hidden_sizes[-1]).cuda()
        else:
            eps = torch.randn(self.batch_size, self.hidden_sizes[-1])
	
        return mu + torch.exp(log_var / 2) * eps
    
    def forward(self, x, code, struc=None):
        
        ###########
        # Encoder #
        ###########
        
        # get the code from the tensor
        # add the conditioned code
        if args_dict["dataset"]!="struc":
            x = torch.cat((x,code),1)
        else:
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
        if args_dict["dataset"]!="struc": 
            sample = torch.cat((sample, code),1)
        else:
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




# =============================================================================
# Training
# =============================================================================

# init the networks
if cuda:
    ff = feed_forward(X_dim, hidden_size, batch_size).cuda()
else:
    ff = feed_forward(X_dim, hidden_size, batch_size)

# change the loading bit here
if load:
    ff.load_state_dict(torch.load("models/metal16_nostruc", map_location=lambda storage, loc: storage))


# Loss and Optimizer
solver = optim.Adam(ff.parameters(), lr=lr)
burn_in_counter = 0
tick = 0


# number of epochs
num_epochs=args_dict['num_epochs']

if train:
    for its in range(num_epochs):
        
        #############################
        # TRAINING 
        #############################
        
        ff.train()
        scores=[]
        data=shuffle(data)
        print("Grammar Cond. - Epoch: {0}/{1}  Latent: {2}".format(its,num_epochs,hidden_size[-1]))
        for it in range(n // batch_size):
        
            if args_dict["dataset"]=="nostruc":
                
                x_batch=data[it * batch_size: (it + 1) * batch_size]
                code = x_batch[:,-8:]
                x_batch = x_batch[:,:3080]

                if cuda:
                    X = torch.from_numpy(x_batch).cuda().type(torch.cuda.FloatTensor)
                    C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
                else:
                    X = torch.from_numpy(x_batch).type(torch.FloatTensor)
                    C = torch.from_numpy(code).type(torch.FloatTensor)

                
            else:
                x_batch=data[it * batch_size: (it + 1) * batch_size]
                code = x_batch[:,-8:]
                structure = x_batch[:,3080:-8]
                x_batch = x_batch[:,:3080]

                if cuda:
                    X = torch.from_numpy(x_batch).cuda().type(torch.cuda.FloatTensor)
                    C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
                    S = torch.from_numpy(structure).cuda().type(torch.cuda.FloatTensor) 
                else:
                    X = torch.from_numpy(x_batch).type(torch.FloatTensor)
                    C = torch.from_numpy(code).type(torch.FloatTensor)
                    S = torch.from_numpy(structure).type(torch.FloatTensor)  
    

            
            #turf last gradients
            solver.zero_grad()
            
            
            if args_dict["dataset"]=="struc":
            # Forward
                x_sample, z_mu, z_var = ff(X, C, S)
            else:
                x_sample, z_mu, z_var = ff(X, C)
            
    
                
            # Loss
            recon_loss = nn.binary_cross_entropy(x_sample, X, size_average=False) # by setting to false it sums instead of avg.
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
            #kl_loss=KL_Div(z_mu,z_var,unit_gauss=True,cuda=True)
            kl_loss = kl_loss*burn_in_counter
            loss = recon_loss + kl_loss
            
            
            # Backward
            loss.backward()
        
            # Update
            solver.step()
            
            
            
            len_aa=140*22
            y_label=np.argmax(x_batch[:,:len_aa].reshape(batch_size,-1,22), axis=2)
            y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
            
            
            # can use argmax again for clipping as it uses the first instance of 21
            # loop with 256 examples is only about 3 milliseconds                      
            for idx, row in enumerate(y_label):
                scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
        
        print("Tra Acc: {0}".format(np.mean(scores)))
                
        if its==(num_epochs-1):
            with open('latent_results_'+str(args_dict["dataset"])+'.txt', 'a') as f:
                f.write(str(args_dict['latent_dim'])+' train '+str(np.mean(scores)))


        
        if its>300 and burn_in_counter<1.0:
            burn_in_counter+=0.003
        
        
        
        #############################
        # Validation 
        #############################
        
        scores=[]
        
        ff.eval()
        for it in range(data_test.shape[0] // batch_size):
            x_batch=data_test[it * batch_size: (it + 1) * batch_size]

            if args_dict["dataset"]=="nostruc":

                x_batch=data[it * batch_size: (it + 1) * batch_size]
                code = x_batch[:,-8:]
                x_batch = x_batch[:,:3080]

                if cuda:
                    X = torch.from_numpy(x_batch).cuda().type(torch.cuda.FloatTensor)
                    C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
                else:
                    X = torch.from_numpy(x_batch).type(torch.FloatTensor)
                    C = torch.from_numpy(code).type(torch.FloatTensor)


            else:
                
                x_batch=data[it * batch_size: (it + 1) * batch_size]
                code = x_batch[:,-8:]
                structure = x_batch[:,3080:-8]
                x_batch = x_batch[:,:3080]

                if cuda:
                    X = torch.from_numpy(x_batch).cuda().type(torch.cuda.FloatTensor)
                    C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
                    S = torch.from_numpy(structure).cuda().type(torch.cuda.FloatTensor)
                else:
                    X = torch.from_numpy(x_batch).type(torch.FloatTensor)
                    C = torch.from_numpy(code).type(torch.FloatTensor)
                    S = torch.from_numpy(structure).type(torch.FloatTensor)


            if args_dict["dataset"]=="struc":
            # Forward
                x_sample, z_mu, z_var = ff(X, C, S)
            else:
                x_sample, z_mu, z_var = ff(X, C)

                            

        
            len_aa=140*22
            y_label=np.argmax(x_batch[:,:len_aa].reshape(batch_size,-1,22), axis=2)
            y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)

            for idx, row in enumerate(y_label):
                scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
        print("Val Acc: {0}".format(np.mean(scores)))

        if its==(num_epochs-1):
            with open('latent_results_'+str(args_dict["dataset"])+'.txt', 'a') as f:
                f.write(str(args_dict['latent_dim'])+' test '+str(np.mean(scores)))



# saves if its running on gpu          
if cuda:
    torch.save(ff.state_dict(), 'metal'+str(args_dict['latent_dim'])+"_"+str(args_dict['dataset']))



def newMetalBinder(model,data,name):
    """
    Generates a new sequence based on a metal code and a grammar. 
    The data array is (4353,) where the first 3080 are the
    sequence, the next 1265 are the fold and the final 8 are the metal
    binding flags. Fold is optional
    """
    scores=[]
    #model into eval mode    
    model.eval()
    
    if args_dict["dataset"]=="nostruc":

        code = np.tile(data[-8:],(model.batch_size,1))
        x = np.tile(data[:3080],(model.batch_size,1))

        if cuda:
            X = torch.from_numpy(x).cuda().type(torch.cuda.FloatTensor)
            C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
        else:
            X = torch.from_numpy(x).type(torch.FloatTensor)
            C = torch.from_numpy(code).type(torch.FloatTensor)


    else:
        
        code = np.tile(data[-8:],(model.batch_size,1))
        structure = np.tile(data[3080:-8],(model.batch_size,1))
        x = np.tile(data[:3080],(model.batch_size,1))

        if cuda:
            X = torch.from_numpy(x).cuda().type(torch.cuda.FloatTensor)
            C = torch.from_numpy(code).cuda().type(torch.cuda.FloatTensor)
            S = torch.from_numpy(structure).cuda().type(torch.cuda.FloatTensor)
        else:
            X = torch.from_numpy(x).type(torch.FloatTensor)
            C = torch.from_numpy(code).type(torch.FloatTensor)
            S = torch.from_numpy(structure).type(torch.FloatTensor)


    if args_dict["dataset"]=="struc":
    # Forward
        x_sample, z_mu, z_var = ff(X, C, S)
    else:
        x_sample, z_mu, z_var = ff(X, C)
    
    
    len_aa=140*22
    y_label=np.argmax(x[:,:len_aa].reshape(batch_size,-1,22), axis=2)
    y_pred =np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    #np.save(name,y_pred)
    print(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22).shape)
    np.save(name,x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22))
    
    
    for idx, row in enumerate(y_label):
        scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Accuracy: {0}".format(np.mean(scores)))
    
    return


if new_metal:
    name="prots_nomet"
    g1=np.load(name+".npy")
    g1=g1[1]
    if len(g1.shape)<2:
        newMetalBinder(ff,g1,name+"_out")
    else:
            
        for idx, row in enumerate(g1):   
            newMetalBinder(ff,row,name+"_out_"+str(idx))

               
