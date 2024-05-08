import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, z_dim=1):
        super().__init__()

        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(118,60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, z_dim)
        )
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, 118)
        )

    def forward(self,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMM, self).__init__()
        #Encoder network
        self.fc1 = nn.Linear(78, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4_mean = nn.Linear(10, z_dim)
        
        # 对数方差的编码器网络
        self.fc4_logvar = nn.Linear(10,z_dim)


        #Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 78)

        #Estimation network
        self.fc9 = nn.Linear(z_dim+2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4_mean(h), self.fc4_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        # z_c = self.encode(x)
        z_c = self.reparameterize(mu,logvar)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return mu, logvar,x_hat, z_c, gamma, z

class BetaVAE(nn.Module):
    def __init__(self, z_dim=1, beta=1.0, n_gmm=2):
        super(BetaVAE, self).__init__()
        self.beta = beta
        # Reuse the encoder and decoder definitions from the VAE
        self.encoder = nn.Sequential(
            nn.Linear(118, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, z_dim*2)  # Output both mu and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, 118)
        )
        self.estimation = nn.Sequential(
            nn.Linear(z_dim + 2, 10),
            nn.Tanh(),
            nn.Linear(10, n_gmm * 3)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        outputs = self.encoder(x)
        mu = outputs[:, :outputs.shape[1]//2]
        logvar = outputs[:, outputs.shape[1]//2:]
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        gmm_params = self.estimation(torch.cat((z, x_hat), dim=1))
        return x_hat, mu, logvar, self.beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())), gmm_params
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity


class VAE(nn.Module):
    def __init__(self, z_dim=8):
        super(VAE, self).__init__()
        # Encoder network
        self.fc1 = nn.Linear(78, 60)  # Assuming input feature size is 78
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4_mean = nn.Linear(10, z_dim)  # Output mean of z
        self.fc4_logvar = nn.Linear(10, z_dim)  # Output log variance of z

        # Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 78)  # Assuming output feature size is the same as input

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4_mean(h), self.fc4_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        h = F.relu(self.fc5(x))
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return self.fc8(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return mu, logvar, x_hat
    

