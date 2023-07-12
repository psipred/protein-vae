#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:53:06 2017

@author: Lewis Moffat aka. Groovy Dragon

This script trains a conditional variational autoencoder in pytorch.

The two datasets this can be trained on are assembled_data_mbflip.npy
and assembled_data_mbflip_fold.npy. They are called nostruc and struc
respectively. The struc set contains 1265 extra dimensions per datum that
describes the fold grammar. Either can be used but it needs to be specified in
input args.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

import vae
from utils import load_data


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, condition_size, batch_size=1):
        super().__init__()

        self.input_size = input_size

        self.hidden_sizes = hidden_sizes
        self.condition_size = condition_size
        self.latent_size = hidden_sizes[-1]
        self.batch_size = batch_size

        self.x_size = input_size - condition_size
        self.code_size = 8  # hard coded
        self.struc_size = condition_size - self.code_size

        self.fc = nn.Linear(input_size, hidden_sizes[0])
        self.bn = nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = nn.Linear(hidden_sizes[2], hidden_sizes[3])

        self.fc4 = nn.Linear(hidden_sizes[3] + condition_size, hidden_sizes[2])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.bn5 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.bn6 = nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = nn.Linear(hidden_sizes[0], input_size - condition_size)

    @property
    def device(self):
        return next(p.device for p in self.fc.parameters())

    def _sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a Gaussian
        return mu + torch.exp(log_var / 2) * Variable(torch.randn_like(mu))

    def condition(self, x, code, struc=None):
        return torch.cat((x, code) if struc is None else (x, code, struc), -1)

    def forward(self, x, code, struc=None):
        mu, sig = self.encoder(x, code, struc)
        sig = F.softplus(sig)  # This is a bug!!!
        z = self._sample_z(mu, sig)
        out = self.decoder(z, code, struc)
        return out, mu, sig

    def generator(self, code, struc=None, num_samples=1):
        code = torch.as_tensor(code, device=self.device).float()
        code = code.reshape(-1, self.code_size).tile(num_samples, 1)
        if struc is not None:
            struc = torch.as_tensor(struc, device=self.device).float()
            struc = struc.reshape(-1, self.struc_size).tile(num_samples, 1)
        z = torch.randn(code.shape[0], self.latent_size, device=code.device)
        return self.decoder(z, code, struc)

    def encoder(self, x, code, struc=None):
        x = self.condition(x, code, struc)
        out1 = self.bn(self.fc(x)).relu()
        out2 = self.bn1(self.fc1(out1)).relu()
        out3 = self.bn2(self.fc2(out2)).relu()
        mu = self.fc3_mu(out3)
        sig = self.fc3_sig(out3)
        return mu, sig

    def decoder(self, z, code, struc=None):
        z = self.condition(z, code, struc)
        out4 = self.bn4(self.fc4(z)).relu()
        out5 = self.bn5(self.fc5(out4)).relu()
        out6 = self.bn6(self.fc6(out5)).relu()
        out7 = self.fc7(out6).sigmoid()
        return out7

    def extract_label(self, x):
        return x.reshape(x.shape[0], -1, 22).argmax(-1)

    def compute_scores(self, y_pred, y_label):
        scores = [
            matches[:label_len].mean()
            for matches, label_len in zip(1.0 * (y_pred == y_label), y_label.argmax(-1))
        ]
        return np.array(scores)


class Dataset(TensorDataset):
    def __getitem__(self, index):
        return tuple(tensor[index].float() for tensor in self.tensors)


def make_autoencoder(struc, latent_dim, load_path=None):
    model = VariationalAutoEncoder(
        input_size=4353 if struc else 3088,
        hidden_sizes=[512, 256, 128, latent_dim],
        condition_size=1273 if struc else 8,
    )
    if load_path:
        state_dict = torch.load(load_path, map_location=lambda storage, _: storage)
        state_dict = {k.lower(): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model.eval()


def make_dataset(data, struc=False):
    data = torch.tensor(data)
    x = data[:, :3080]
    code = data[:, -8:]
    if struc:
        struc = data[:, 3080:-8]
        return Dataset(x, code, struc)
    return Dataset(x, code)


def train(model, data, args):
    num_epochs = args.num_epochs
    dataset = make_dataset(data, args.struc)
    rng = torch.Generator().manual_seed(0)
    train_size = round(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, lengths=[train_size, test_size], generator=rng)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in tqdm(range(num_epochs), "Epochs"):
        kl_loss_scale = np.clip((epoch - 300) * 0.003, 0.0, 1.0)
        model.train()
        scores_train = []
        for batch in tqdm(train_loader, f"Training...", leave=False):
            x = batch[0]
            x_sample, z_mu, z_var = model(*batch)

            # Loss
            recon_loss = F.binary_cross_entropy(x_sample, x, reduction="sum")
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
            loss = recon_loss + kl_loss * kl_loss_scale

            # Optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_label = model.extract_label(x)
            y_pred = model.extract_label(x_sample)
            scores_train.extend(model.compute_scores(y_pred, y_label))

        scores_val = []
        model.eval()
        for batch in tqdm(test_loader, f"Validating...", leave=False):
            x = batch[0]
            x_sample, z_mu, z_var = model(*batch)
            y_label = model.extract_label(x)
            y_pred = model.extract_label(x_sample)
            scores_val.extend(model.compute_scores(y_pred, y_label))

        acc_train, acc_val = (np.mean(scores) for scores in (scores_train, scores_val))
        log_str = f"Epoch {epoch}:\tTra Acc: {acc_train:.5%}\tVal Acc: {acc_val:.5%}"
        model_name = "struc" if model.struc_size > 0 else "nostruc"
        log_path = f"logs/latent_results_{model_name}_{model.latent_size}.txt"
        print(log_str)
        with open(log_path, "a") as f:
            print(log_str, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--struc", action="store_true", help="Condition on encoded grammar")
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--data", type=str, help="Path to data", default="data/assembled_data_fold_csr.npz")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force cpu mode")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    data = load_data(args.data)
    model = vae.make_autoencoder(args.struc, args.latent_dim, args.load_path).to(device)
    train(model, data, args)
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
