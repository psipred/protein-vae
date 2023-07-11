import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, condition_size, batch_size):
        super().__init__()

        self.input_size = input_size

        self.hidden_sizes = hidden_sizes
        self.condition_size = condition_size
        self.latent_size = hidden_sizes[-1]
        self.batch_size = batch_size

        self.fc = nn.Linear(input_size, hidden_sizes[0])  # 2 for bidirection
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

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a Gaussian
        return mu + torch.exp(log_var / 2) * Variable(torch.randn_like(mu))

    def condition(self, x, code, struc=None):
        return torch.cat((x, code) if struc is None else (x, code, struc), -1)

    def forward(self, x, code, struc=None):
        mu, sig = self.encoder(x, code, struc)
        sig = F.softplus(sig)
        z = self.sample_z(mu, sig)
        out = self.decoder(z, code, struc)
        return out, mu, sig

    def generator(self, code, struc=None):
        z = torch.randn(self.batch_size, self.latent_size, device=code.device)
        return self.decoder(z, code, struc)

    def encoder(self, x, code, struc=None):
        x = self.condition(x, code, struc)
        out1 = F.relu(self.bn(self.fc(x)))
        out2 = F.relu(self.bn1(self.fc1(out1)))
        out3 = F.relu(self.bn2(self.fc2(out2)))
        mu = self.fc3_mu(out3)
        sig = self.fc3_sig(out3)
        return mu, sig

    def decoder(self, z, code, struc=None):
        z = self.condition(z, code, struc)
        out4 = F.relu(self.bn4(self.fc4(z)))
        out5 = F.relu(self.bn5(self.fc5(out4)))
        out6 = F.relu(self.bn6(self.fc6(out5)))
        out7 = F.sigmoid(self.fc7(out6))
        return out7
