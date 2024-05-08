import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.metrics import silhouette_score
from forward_step import ComputeLoss

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def evalAE(model, dataloaders, device):
    """Testing the Autoencoder model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Testing...')

    reconstruction_errors = []
    labels_total = []

    # Evaluate on both training and test data
    for phase, loader in [('train', dataloader_train), ('test', dataloader_test)]:
        for x, y in loader:
            # print(f"x = {x}")
            # print(f"y = {y}")
            x = x.float().to(device)
            x_hat = model(x)
            # Compute relative Euclidean distance and detach before converting to numpy
            rec_error = (torch.norm(x - x_hat, dim=1) / torch.norm(x, dim=1)).detach().cpu().numpy()
            reconstruction_errors.append(rec_error)
            labels_total.append(y.cpu().numpy())

    # Flatten lists
    reconstruction_errors = np.concatenate(reconstruction_errors)
    labels_total = np.concatenate(labels_total)

    # Compute threshold based on the training set
    threshold = np.percentile(reconstruction_errors, 95)  # Example threshold

    # Make predictions based on the threshold
    predictions = (reconstruction_errors > threshold).astype(int)

    # Calculate performance metrics
    precision, recall, f_score, _ = prf(labels_total, predictions, average='weighted')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

    # ROC AUC
    # if np.any(labels_total):
    #     roc_auc = roc_auc_score(labels_total, reconstruction_errors)
    #     print('ROC AUC score:', roc_auc)

    return labels_total, reconstruction_errors, precision, recall, f_score

def evalVAE(model, dataloaders, device):
    """Evaluates the Variational Autoencoder model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Evaluating VAE model...HERE')

    reconstruction_errors = []
    labels_total = []

    # Evaluate on both training and test data
    for phase, loader in [('train', dataloader_train), ('test', dataloader_test)]:
        for x, y in loader:
            x = x.float().to(device)
            
            # Forward pass through VAE model
            mu, logvar, x_hat = model(x)
            
            # Compute relative Euclidean distance and detach before converting to numpy
            rec_error = (torch.norm(x - x_hat, dim=1) / torch.norm(x, dim=1)).detach().cpu().numpy()
            reconstruction_errors.append(rec_error)
            labels_total.append(y.cpu().numpy())

    # Flatten lists
    reconstruction_errors = np.concatenate(reconstruction_errors)
    labels_total = np.concatenate(labels_total)

    # Compute threshold based on the training set
    threshold = np.percentile(reconstruction_errors, 95)  # Example threshold

    # Make predictions based on the threshold
    predictions = (reconstruction_errors > threshold).astype(int)

    # Calculate performance metrics
    precision, recall, f_score, _ = prf(labels_total, predictions, average='weighted')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")
    accuracy = accuracy_score(labels_total, predictions)
    print(f"Acc = {accuracy:.4f}, and labels_total = {labels_total}")
    # # ROC AUC
    # if np.any(labels_total):
    #     roc_auc = roc_auc_score(labels_total, reconstruction_errors, multi_class='ovo')
    #     print('ROC AUC score:', roc_auc)

    return labels_total, reconstruction_errors, precision, recall, f_score

def evalGMM(model, dataloaders, device):
    """Evaluates the Gaussian Mixture Model"""
    dataloader_train, dataloader_test = dataloaders
    print('Evaluating GMM model...')
    model.to(device)

    labels_total = []
    scores_total = []
    log_prob_total = []

    # Evaluate on both training and test data
    for phase, loader in [('train', dataloader_train), ('test', dataloader_test)]:
        for x, y in loader:
            x = x.float().numpy()  # Make sure data is in NumPy format for scikit-learn
            
            # Forward pass through GMM model
            labels = model.predict(x)
            log_probs = model.score_samples(x)
            scores_total.extend(log_probs)
            labels_total.extend(labels)
            log_prob_total.append(log_probs)

    # Flatten lists
    labels_total = np.array(labels_total)
    scores_total = np.array(scores_total)
    log_prob_total = np.concatenate(log_prob_total)

    # Compute silhouette score on the last batch
    silhouette_avg = silhouette_score(x, labels, metric='euclidean')
    print(f"GMM Silhouette Score: {silhouette_avg:.2f}")

    # Make predictions based on the log probabilities
    threshold = np.percentile(log_prob_total, 5)  # Lower 5% as anomaly
    predictions = (log_prob_total < threshold).astype(int)

    # Calculate performance metrics
    precision, recall, f_score, _ = prf(y, predictions, average='weighted')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

    accuracy = accuracy_score(labels_total, predictions)
    print(f"Acc = {accuracy:.4f}, and labels_total = {labels_total}")

    return labels_total, scores_total, silhouette_avg, precision, recall, f_score


def evalVAE1(model, dataloaders, device, n_gmm):
    """Testing the DAGMM model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, None, None, device, n_gmm)
    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
        for x, _ in dataloader_train:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            
            N_samples += x.size(0)
            
        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # Obtaining Labels and energy scores for train data
        energy_train = []
        labels_train = []
        for x, y in dataloader_train:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, phi=train_phi,
                                                              mu=train_mu, cov=train_cov, 
                                                              sample_mean=False)
            
            energy_train.append(sample_energy.detach().cpu())
            labels_train.append(y)
        energy_train = torch.cat(energy_train).numpy()
        labels_train = torch.cat(labels_train).numpy()

        # Obtaining Labels and energy scores for test data
        energy_test = []
        labels_test = []
        for x, y in dataloader_test:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
                                                              train_mu, train_cov,
                                                              sample_mean=False)
            
            energy_test.append(sample_energy.detach().cpu())
            labels_test.append(y)
        energy_test = torch.cat(energy_test).numpy()
        labels_test = torch.cat(labels_test).numpy()
    
        scores_total = np.concatenate((energy_train, energy_test), axis=0)
        labels_total = np.concatenate((labels_train, labels_test), axis=0)

    threshold = np.percentile(scores_total, 100 - 20)
    pred = (energy_test > threshold).astype(int)
    gt = labels_test.astype(int)
    precision, recall, f_score, _ = prf(gt, pred, average='binary')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total)*100))
    return labels_total, scores_total


def eval(model, dataloaders, device, n_gmm):
    """Testing the DAGMM model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, None, None, device, n_gmm)
    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
        for x, _ in dataloader_train:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            
            N_samples += x.size(0)
            
        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # Obtaining Labels and energy scores for train data
        energy_train = []
        labels_train = []
        for x, y in dataloader_train:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, phi=train_phi,
                                                              mu=train_mu, cov=train_cov, 
                                                              sample_mean=False)
            
            energy_train.append(sample_energy.detach().cpu())
            labels_train.append(y)
        energy_train = torch.cat(energy_train).numpy()
        labels_train = torch.cat(labels_train).numpy()

        # Obtaining Labels and energy scores for test data
        energy_test = []
        labels_test = []
        for x, y in dataloader_test:
            x = x.float().to(device)

            mu, logvar, _, _, z, gamma = model(x)
            sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
                                                              train_mu, train_cov,
                                                              sample_mean=False)
            
            energy_test.append(sample_energy.detach().cpu())
            labels_test.append(y)
        energy_test = torch.cat(energy_test).numpy()
        labels_test = torch.cat(labels_test).numpy()
    
        scores_total = np.concatenate((energy_train, energy_test), axis=0)
        labels_total = np.concatenate((labels_train, labels_test), axis=0)

    threshold = np.percentile(scores_total, 100 - 20)
    pred = (energy_test > threshold).astype(int)
    gt = labels_test.astype(int)
    precision, recall, f_score, _ = prf(gt, pred, average='binary')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total)*100))
    return labels_total, scores_total