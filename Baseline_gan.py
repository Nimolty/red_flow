# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:56:54 2022

@author: lenovo
"""
import pickle
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import numpy as np
import time
import argparse
import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from RLEnvSortingBall_new import RLSorting
from utils_new import exists_or_mkdir, images_to_video, visualize_states
from GAN_part import sample_noise, discriminator, generator, discriminator_loss, generator_loss, get_optimizer, run_and_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type = str, default='GAN_Ball')
    parser.add_argument('--epoches', type=int, default=1000)
    parser.add_argument('--sample_num', type=int, default=100000)
    parser.add_argument('--output_num', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--n_boxes', type=int, default=15)
    parser.add_argument('--wall_bound', type=float, default=0.3)
    parser.add_argument('--max_vel', type=float, default=0.3)
    parser.add_argument('--time_freq',type=int, default=240)
    parser.add_argument('--is_gui', type=bool, default=False)
    
    args = parser.parse_args()
    exp_name = args.exp_name
    sample_num = args.sample_num
    output_num = args.output_num
    batch_size = args.batch_size
    epoches = args.epoches
    num_boxes = args.n_boxes
    wall_bound = args.wall_bound
    max_vel = args.max_vel
    time_freq = args.time_freq
    is_gui = args.is_gui
    suffix = 'image'
    
    exists_or_mkdir(f'./logs')
    tb_path = f'./logs/{exp_name}/tb'
    exists_or_mkdir(tb_path)
    save_path = f'./video/{exp_name}'
    exists_or_mkdir(save_path)
    exists_or_mkdir(f'Images/dynamic_{suffix}')
    
    NOISE_DIM = 24
    STATE_DIM = 30
    
    with open(f'./dataset/exp_0415_name_1.pth', 'rb') as f:
        samples = pickle.load(f)
    writer = SummaryWriter(tb_path)
    
    env_kwargs = {
        'n_boxes' : num_boxes,
        'wall_bound' : wall_bound,
        'max_action' : max_vel,
        'time_freq' : time_freq,
        'is_gui' : is_gui,
        }
    env = RLSorting(**env_kwargs)
    env.reset()

    samples = torch.tensor(samples, dtype = torch.float32)
    targets = torch.tensor([0] * samples.shape[0])
    loader = DataLoader(TensorDataset(samples, targets), batch_size = batch_size)

    G = generator(NOISE_DIM).to(device)
    D = discriminator(STATE_DIM).to(device)
    G_solver = get_optimizer(G)
    D_solver = get_optimizer(D)
    run_and_img(D, G, D_solver, G_solver, discriminator_loss, generator_loss, \
                loader, epoches, writer, env, suffix)
    
    env.close()
    writer.close()