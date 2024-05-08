import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model import DAGMM
from forward_step import ComputeLoss
from forward_step import *
# from utils.utils import weights_init_normal

from utilities.utilities import *
from model import *

import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support as prf
import torch
from torch import nn, optim
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.autograd import Variable


class TrainerAE:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        """Training the AE model"""
        self.model = AE(self.args.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        criterion = torch.nn.MSELoss()
        self.model.train()
        epoch_losses = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                x_hat = self.model(x)  # Get the reconstruction from the model

                loss = criterion(x_hat, x)  # Calculate the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            average_loss = total_loss / len(self.train_loader)
            epoch_losses.append(average_loss)
            if epoch % 5 == 0:
                print(f'Training Autoencoder... Epoch: {epoch}, Loss: {total_loss / len(self.train_loader)}')
        plot_loss(epoch_losses)

class TrainerVAE2:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        # self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
        #                            self.device, self.args.n_gmm)
        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)
    
        self.model.train()
        epoch_losses = []
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                mu, logvar, x_hat, _c, gamma, z = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            average_loss = total_loss / len(self.train_loader)
            epoch_losses.append(average_loss)
            if epoch % 5 == 0:
                print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss/len(self.train_loader)))
        plot_loss(epoch_losses)

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device


    def train(self):
        """Training the DAGMM model"""
        # if self.args.model == 'vae':
        #     self.model = VAE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'ae':
        #     self.model = AE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'betavae':
        #     self.model = BetaVAE(self.args.latent_dim, beta=self.args.beta).to(self.device)

        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        # self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
        #                            self.device, self.args.n_gmm)
        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)
    
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                mu, logvar, x_hat, _c, gamma, z = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
                

class TrainerBetaVAEwEstimationNetWork:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device


    def train(self):
        """Training the DAGMM model"""
        # if self.args.model == 'vae':
        #     self.model = VAE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'ae':
        #     self.model = AE(self.args.latent_dim).to(self.device)
        # elif self.args.model == 'betavae':
        self.model = BetaVAE(self.args.latent_dim, beta=self.args.beta).to(self.device)

        # self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        # self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
        #                            self.device, self.args.n_gmm)
        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)
    
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()
                
                mu, logvar, x_hat, _c, gamma, z = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))


class TrainerGMM:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.n_components = args.n_gmm
        self.phi = None
        self.mu = None
        self.cov = None
        self.num_epochs = args.num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            for X_train, _ in self.train_loader:
                X_train = X_train.to(self.device)

                # Randomly initialize gamma for each epoch
                gamma = torch.rand((X_train.size(0), self.n_components), device=self.device)
                gamma /= gamma.sum(dim=1, keepdim=True)

                # Compute GMM parameters
                self.phi, self.mu, self.cov = self.compute_gmm_params(X_train, gamma)
            print(f"GMM training completed for Epoch {epoch + 1}.")

    def compute_gmm_params(self, z, gamma):
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / gamma.size(0)
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        return phi, mu, cov

    def compute_energy(self, z):
        if self.phi is None or self.mu is None or self.cov is None:
            raise RuntimeError("GMM parameters have not been initialized.")
        
        z_mu = z.unsqueeze(1) - self.mu.unsqueeze(0)
        cov_inverse = []
        det_cov = []
        eps = 1e-3
        for i in range(self.n_components):
            cov_k = self.cov[i] + torch.eye(self.cov[i].size(-1), device=self.device) * eps
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(torch.det(cov_k * (2 * np.pi)).item())
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.tensor(det_cov, dtype=torch.float32, device=self.device)
        exp_term_tmp = -0.5 * torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=[-2, -1])
        max_val = torch.max(exp_term_tmp, dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(self.phi.unsqueeze(0) * exp_term / torch.sqrt(det_cov).unsqueeze(0), dim=1) + eps)
        return torch.mean(sample_energy), torch.sum(1 / cov_inverse.diag(dim=-1), dim=1).mean()

    def evaluate(self):
        self.to(self.device)  # Ensure model components are on the correct device
        print('Evaluating GMM...')
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X_test, y_true in self.test_loader:
                X_test = X_test.to(self.device)
                y_true = y_true.numpy()  # Assume labels are already numpy arrays for metric calculation
                predicted_labels = self.predict(X_test)  # Implement your prediction method based on phi, mu, cov
                all_labels.extend(y_true)
                all_predictions.extend(predicted_labels.cpu().numpy())

        precision, recall, f_score, _ = prf(all_labels, all_predictions, average='weighted')
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Acc = {accuracy:.4f}")

    def predict(self, x):
        # This should be an actual implementation based on the GMM parameters
        # Here's a dummy implementation:
        return torch.randint(0, self.n_components, (x.size(0),), device=self.device)

    def to(self, device):
        # If phi, mu, and cov are tensors, move them to the specified device
        if self.phi is not None:
            self.phi = self.phi.to(device)
        if self.mu is not None:
            self.mu = self.mu.to(device)
        if self.cov is not None:
            self.cov = self.cov.to(device)
        return self
    # def evaluate(self):
    #     # 从 DataLoader 中获取测试数据
    #     print("Evaluating GMM...")
    #     X_test, _ = next(iter(self.test_loader))
    #     X_test = X_test.to(self.device)  # 转换为适合的设备

    #     # 计算能量
    #     energy, _ = self.compute_energy(X_test)
    #     print(f"Average energy: {energy.item():.2f}")





class TrainerGMM1:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.n_components = args.n_gmm  # GMM 的组件数
        self.covariance_type = args.covtype
        self.tol = args.tol
        self.model = GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type, tol = self.tol,  random_state=42)

    def train(self):
        # 从 DataLoader 中获取数据
        X_train, y_train = next(iter(self.train_loader))
        X_train = X_train.numpy()  # 转换为 NumPy 数组

        # 训练 GMM
        self.model.fit(X_train)
        print("GMM training completed.")

    def evaluate(self):
        # 从 DataLoader 中获取测试数据
        print("Evaluating GMM...")

        X_test, y_test = next(iter(self.test_loader))
        X_test = X_test.numpy()  # 转换为 NumPy 数组

        # 预测并计算指标
        labels = self.model.predict(X_test)
        scores = self.model.score_samples(X_test)
        silhouette_avg = silhouette_score(X_test, labels)
        print(f"GMM Silhouette Score: {silhouette_avg:.2f}")
        
        real_distribution = np.random.normal(loc=np.mean(scores), scale=np.std(scores), size=len(scores))

        return labels, scores, silhouette_avg, real_distribution

    def plot_results(self, labels, scores, real_distribution=None):
        # # 画出结果，例如分数或者其他指标
        plt.scatter(range(len(scores)), scores, c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar()
        plt.xlabel('Sample index')
        plt.ylabel('Score (log probability)')
        plt.title('GMM Scores by Sample')
        plt.show()

def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0]  # 假设所有的子列表长度一样，取第一个元素就好
    return shape

class TrainerVAEGMM:
    """Trainer class for VAEGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.compute = ComputeLoss1(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.args.lambda_recon, self.args.lambda_kl, self.device, self.args.n_gmm)

    def train(self):
        """Training the VAEGMM model"""
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                self.optimizer.zero_grad()

                mu, logvar, x_hat, z_c, gamma, z = self.model(x)
                loss = self.compute.forward(x, x_hat, z, gamma, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                total_loss += loss.item()
            if epoch % 5 == 0:
                print(f'Training VAEGMM... Epoch: {epoch}, Loss: {total_loss / len(self.train_loader)}')

    # def update_gmm(self, z):
    #     self.gmm.fit(z)
    def evaluate(self):
        """Evaluates the VAEGMM model on the test set."""
        self.model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.float().to(self.device)
                _, _, _, _, z, gamma = self.model(x)
                sample_energy, _ = self.compute.compute_energy(z, gamma)
                
                scores.extend(sample_energy.cpu().numpy())
                labels.extend(y.numpy())

        # Compute the AUC, precision, recall, and F1-score
        threshold = np.percentile(scores, 100 - 20)
        predictions = (np.array(scores) > threshold).astype(int)
        precision, recall, f_score, _ = prf(labels, predictions, average='binary')
        roc_auc = roc_auc_score(labels, scores)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")
        print(f'ROC AUC score: {roc_auc:.2f}')
    


    # def evaluate(self):
    #         """Evaluate the model on the test set."""
    #         self.model.eval()
    #         true_labels = []
    #         predicted_scores = []
    #         predicted_labels = []

    #         with torch.no_grad():
    #             for x, labels in self.test_loader:
    #                 x = x.float().to(self.device)
    #                 mu, logvar, x_hat, z_c, gamma, z = self.model(x)
    #                 probabilities = torch.softmax(gamma, dim=1)  # Convert to probabilities
    #                 scores = probabilities[:, 1]  # Assuming the second column is for the positive class
    #                 preds = probabilities.argmax(dim=1)

    #                 true_labels.extend(labels.cpu().tolist())
    #                 predicted_scores.extend(scores.cpu().tolist())
    #                 predicted_labels.extend(preds.cpu().tolist())

    #         # Calculate metrics
    #         print(f"true_labels = {true_labels}")
    #         print(f"predicted_scores = {predicted_scores}")
    #         print(f"true_labels = {get_shape(true_labels)}")
    #         print(f"predicted_scores = {get_shape(predicted_scores)}")
    #         true_labels = np.array(true_labels)
    #         predicted_scores = np.array(predicted_scores)
    #         # roc_auc = roc_auc_score(true_labels, predicted_scores, multi_class="ovo", average='macro')
    #         precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

    #         # print(f"ROC AUC: {roc_auc:.3f}")
    #         print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def plot_results(self, labels, scores, real_distribution):
        plt.scatter(range(len(scores)), scores, c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar()
        plt.xlabel('Sample Index')
        plt.ylabel('Score (log probability)')
        plt.title('GMM Scores by Sample')
        plt.show()


class TrainerVAE:
    """Trainer class for VAE without GMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
        self.model = VAE(self.args.latent_dim).to(self.device)  # Assuming VAE is defined somewhere
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.epoch_losses = []
        self.test_losses = []

        self.test = []

                # Metrics
        # Metrics initialization with task type
        self.accuracy = Accuracy(task='multiclass',num_classes=15).to(device)  # Adjust num_classes as necessary
        # self.precision = Precision(task='multiclass',num_classes=15, average='macro').to(device)
        # self.recall = Recall(task='multiclass',num_classes=15, average='macro').to(device)
        # self.f1 = F1Score(task='multiclass',num_classes=15, average='macro').to(device)

    def train(self):
        """Training the VAE model"""
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)
                self.optimizer.zero_grad()

                mu, logvar, x_hat = self.model(x)  # Ensure your VAE model returns these
                recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.99 * kld_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                total_loss += loss.item()
            average_loss = total_loss / len(self.train_loader.dataset)    
            self.epoch_losses.append(average_loss)
            if epoch % 5 == 0:
                print(f'Training VAE... Epoch: {epoch}, Loss: {total_loss / len(self.train_loader)}')

    def evaluate(self):
        """Evaluates the VAE model on the test set."""
        self.model.eval()
        total_loss = 0
        self.accuracy.reset()
        # self.precision.reset()
        # self.recall.reset()
        # self.f1.reset()

        with torch.no_grad():
            for x, labels in self.test_loader:
                x = x.float().to(self.device)
                labels = labels.to(self.device)

                mu, logvar, x_hat = self.model(x)
                recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                total_loss += loss.item()

                # Assuming your model's output is suitable for classification
                preds = torch.argmax(x_hat, dim=1)
                self.test.append(preds)
                self.accuracy.update(preds, labels)
                # self.precision.update(preds, labels)
                # self.recall.update(preds, labels)
                # self.f1.update(preds, labels)
        print(f"test v= {self.test}")
        average_loss = total_loss / len(self.test_loader.dataset)
        print(f'Test Loss: {average_loss:.4f}')
        print(f'Accuracy: {self.accuracy.compute():.4f}')
        # print(f'Precision: {self.precision.compute():.4f}')
        # print(f'Recall: {self.recall.compute():.4f}')
        # print(f'F1 Score: {self.f1.compute():.4f}')

        return average_loss

        # # Calculate metrics
        # threshold = np.percentile(all_preds, 80)  # Set threshold
        # predictions = (all_preds > threshold).astype(int)
        # precision, recall, f_score, _ = prf(all_labels, predictions, average='binary')
        # roc_auc = roc_auc_score(all_labels, all_preds)

        # print(f'ROC AUC: {roc_auc:.4f}')
        # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f_score:.4f}')

        # Calculate additional metrics if needed:
        # e.g., ROC AUC, precision, recall, etc. based on the specific task
        # For now, just show the average test loss as an output.

        return average_loss
    def plot_loss(self):
        """Plot the training loss per epoch."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_test_loss(self):
        """Plot the test loss for each batch."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.test_losses, label='Test Loss per Batch')
        plt.title('Test Loss Over Batches')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()