# code based on https://github.com/danieltan07

from load_data import *
from train import *
from model import *
from test import *
import os
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import numpy as np
import argparse 
import torch

from train import TrainerDAGMM
from test import eval
from preprocess import get_KDDCup99




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.01,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    # Model Selection
    parser.add_argument('--model', type=str, default='gmm',
                        help='Choices: ae OR vae OR betavae')
    # Datasets
    parser.add_argument('--dataset', type=str, default='cicids',
                        help='Choices: kdd OR cicids')
    # GMM
    parser.add_argument('--covtype', type=str, default='spherical',
                        help='Choices: full, tied, diag, and spherical')
    parser.add_argument('--tol', type=int, default=1e-6,
                        help='Model precision')
    # VAE
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Weight for the reconstruction loss in the VAE.')
    parser.add_argument('--lambda_kl', type=float, default=0.5,
                        help='Weight for the KL divergence loss in the VAE.')
    # Beta-VAE
    parser.add_argument('--lambda_beta', type=float, default=0.01,
                        help='Weight for the reconstruction loss in the Beta-VAE.')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get train and test dataloaders.
    # data = get_KDDCup99(args)
    data = Get_Data(args)
    print(f"data = {data}")
    # sys.exit()
    
    if args.model == 'ae':
        AE = TrainerAE(args, data, device)
        AE.train()
        print(f"ae")
        labels, scores, precision, recall, f_score = evalAE(AE.model, data, device)
        # plot_metrics(labels, scores)
    elif args.model == 'vae':
        # VAE = TrainerVAE(args, data, device)
        # VAE.train()
        # labels, scores = evalVAE(VAE.model, data, device, args.n_gmm)
        # print(f"VAE")
        VAE = TrainerVAE(args, data, device)
        VAE.train()
        labels, scores, precision, recall, f_score = evalVAE(VAE.model, data, device)
        # average_loss = VAE.evaluate()
        # VAE.plot_loss()
        # VAE.plot_test_loss()
    elif args.model == 'betavae':
        pass
    elif args.model == 'gmm':
        GMM = TrainerGMM(args, data, device)
        GMM.train()
        # labels, scores, precision, recall, f_score, test = evalGMM(GMM.model, data, device)
        GMM.evaluate()
        # GMM.plot_results(labels, scores,real_distribution)
        # GMM.plot_results_3d(labels, scores)
    elif args.model == 'vaegmm':
        VAEGMM = TrainerVAEGMM(args,data , device)
        VAEGMM.train()
        labels, scores, silhouette_avg, real_distribution = VAEGMM.evaluate()
        VAEGMM.plot_results(labels, scores, real_distribution)


    # DAGMM = TrainerDAGMM(args, data, device)
    # DAGMM.train()
    # DAGMM.eval(DAGMM.model, data, device) # data[1]: test dataloader
    
    # labels, scores = eval(DAGMM.model, data, device, args.n_gmm)


    # BetaVAE


    