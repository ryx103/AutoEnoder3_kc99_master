import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, lambda_recon, lambda_kl, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma, mu, logvar):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))
        # print(f"forward HERE")

        #KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        # loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        loss = (self.lambda_recon * reconst_loss +
                self.lambda_kl * kld_loss +
                self.lambda_energy * sample_energy +
                self.lambda_cov * cov_diag)
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # eps = 1e-12
        eps = 1e-6
        # cov_k += torch.eye(cov_k.size(0)) * eps
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        # print("n_gmm:", self.n_gmm)
        # print("cov size:", cov.size(0))

        # for k in range(self.n_gmm):
        for k in range(cov.size(0)):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)


        z_mu = z_mu.unsqueeze(-1)  # Shape becomes [batch_size, n_gmm, D, 1]
        cov_inverse = cov_inverse.unsqueeze(0)  # Shape becomes [1, n_gmm, D, D]
        # print("z_mu shape:", z_mu.shape)
        # print("cov_inverse shape:", cov_inverse.shape)


        E_z = -0.5 * torch.sum(torch.sum(z_mu * cov_inverse, dim=-2) * z_mu.squeeze(-1), dim=-1)

        # E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean:
            E_z = torch.mean(E_z)
        return E_z, cov_diag


    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        # mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        # mu /= torch.sum(gamma, dim=0).unsqueeze(-1)
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0) / torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s



class ComputeLoss1:
    def __init__(self, model, lambda_energy, lambda_cov, lambda_recon, lambda_kl, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma, mu, logvar):
        """Computing the loss function for DAGMM."""
        reconst_loss = F.mse_loss(x_hat, x)  # 更稳定的计算重构损失
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # 归一化 KLD 损失

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        total_loss = (self.lambda_recon * reconst_loss +
                      self.lambda_kl * kld_loss +
                      self.lambda_energy * sample_energy +
                      self.lambda_cov * cov_diag)

        return total_loss  # 不再需要 Variable 包装和 requires_grad=True
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function, handling non-positive definite cases."""
        if phi is None or mu is None or cov is None:
            phi, mu, cov = self.compute_params(z, gamma)

        eps = 1e-2
        cov_k = [cov[i] + torch.eye(cov[i].size(-1), device=self.device) * eps for i in range(cov.shape[0])]

        cov_inverse = []
        det_cov = []
        cov_diag = 0

        for k in range(len(cov_k)):  # Ensure we are iterating over the actual size of cov_k
            L, info = torch.linalg.cholesky_ex(cov_k[k] * (2 * np.pi))
            if info == 0:
                det_cov.append(L.diag().prod().unsqueeze(0))
                cov_inverse_k = torch.inverse(cov_k[k])
                cov_inverse.append(cov_inverse_k.unsqueeze(0))
                cov_diag += torch.sum(1 / cov_k[k].diag())
            else:
                # Handle non-positive definite case by providing default values
                det_cov.append(torch.tensor([1.0], device=self.device).unsqueeze(0))  # Default value
                cov_inverse.append(torch.eye(cov[k].size(-1), device=self.device).unsqueeze(0))

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0)).unsqueeze(-1)
        E_z = -0.5 * torch.sum(torch.sum(z_mu * cov_inverse.unsqueeze(0), dim=-2) * z_mu.squeeze(-1), dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if sample_mean:
            E_z = torch.mean(E_z)

        return E_z, cov_diag

    def compute_params(self, z, gamma):
        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0) / torch.sum(gamma, dim=0).unsqueeze(-1)
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov