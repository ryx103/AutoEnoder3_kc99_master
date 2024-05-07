import os
import argparse

from vae_config import SolverVAE 
from load_data import get_loader
from torch.backends import cudnn
from utils import mkdir, str2bool 

def main(config):
    # For fast training
    cudnn.benchmark = True

    mkdir(config.log_path)
    mkdir(config.model_save_path)

    data_loader = get_loader(config.data_path, batch_size=config.batch_size, mode=config.mode)
    
    solver = SolverVAE(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-3)  
    parser.add_argument('--latent_dim', type=int, default=10, help='Dimensionality of the latent space')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=100)  
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--data_path', type=str, default='dataset.npz')
    parser.add_argument('--log_path', type=str, default='./vae/logs')
    parser.add_argument('--model_save_path', type=str, default='./vae/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)  
    parser.add_argument('--model_save_step', type=int, default=50)  
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)
