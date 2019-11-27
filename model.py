import torch
from torch import nn, optim
from torch.nn import functional as F

class AE(nn.Module):
    def __init__(self,dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z

class VAE(nn.Module):
    def __init__(self,dim):
        super(VAE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

# Original
# class VAE2d(nn.Module):
#     def __init__(self,dim):
#         super(VAE2d, self).__init__()
#         self.dim = dim
#         self.fc1 = nn.Linear(dim, 256)
#         self.fc21 = nn.Linear(256, 20)
#         self.fc22 = nn.Linear(256, 20)
#         self.fc3 = nn.Linear(20, 256)
#         self.fc4 = nn.Linear(256, dim)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.dim))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar, z

class VAE2d(nn.Module):
    def __init__(self,dim):
        super(VAE2d, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc41 = nn.Linear(32, 2)
        self.fc42 = nn.Linear(32, 2)
        self.fc5 = nn.Linear(2, 32)
        self.fc6 = nn.Linear(32, 128)
        self.fc7 = nn.Linear(128, 512)
        self.fc8 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc41(h3), self.fc42(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        return torch.sigmoid(self.fc8(h7))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z