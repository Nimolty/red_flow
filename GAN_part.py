# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:47:42 2022

@author: lenovo
"""
# 训练一个GAN作为baseline，flow这个问题很苦恼，不然先算了
# 这是最trivial的gan的版本，我们需要做出来
import numpy as np
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import PIL
from torch.utils.data import sampler
from tqdm import tqdm, trange
from utils_new import visualize_states

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_DIM = 24
STATE_DIM = 30

def sample_noise(batch_size, dim, seed = None):
    '''
    Generate a pytorch tensor of uniform random noise ;
    random noise uniform in range(-1, 1)
    '''
    if seed is not None:
        torch.manual_seed(seed)
    
    return 2 * torch.rand((batch_size, dim)) - 1

def discriminator(input_dim, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, 64, bias = True),
        nn.LeakyReLU(0.01),
        nn.Linear(64, 64, bias = True),
        nn.LeakyReLU(0.01),
        nn.Linear(64, 1, bias = True)
        )
    return model

def generator(noise_dim = NOISE_DIM, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    model = None
    model = nn.Sequential(
        nn.Linear(noise_dim, 64, bias = True),
        nn.ReLU(),
        nn.Linear(64, 64, bias = True),
        nn.ReLU(),
        nn.Linear(64, 30, bias = True),
        nn.Tanh()
        )
    
    return model

def discriminator_loss(logits_real, logits_fake):
    '''
    Computes the discriminator loss described baove.
    Inputs:
    - logits_real, Tensor of shape (N,) giving scores for the real data
    - logits_fake, Tensor of shape (N,) giving scores for the fake data
    '''
    loss_func = nn.BCEWithLogitsLoss()
    labels_real = torch.ones(logits_real.shape)
    labels_fake = torch.zeros(logits_fake.shape)
    loss = loss_func(logits_real, labels_real) + loss_func(logits_fake, labels_fake)
    return loss

def generator_loss(logits_fake):
    loss_func = nn.BCEWithLogitsLoss()
    labels_fake = torch.ones(logits_fake.shape)
    loss = loss_func(logits_fake, labels_fake)
    return loss

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, betas = (0.5, 0.999))
    return optimizer


G = generator(NOISE_DIM)
D = discriminator(STATE_DIM)
G_solver = get_optimizer(G)
D_solver = get_optimizer(D)

def run_and_img(D, G, D_solver, G_solver, discriminator_loss, generator_loss, \
            loader, num_epoch, writer, env, suffix):
    for epoch in trange(num_epoch):
        for i, sample in enumerate(loader):
            sample = sample[0].to(device)
            D_solver.zero_grad()
            logits_real = D(sample)
            fake_noise = sample_noise(sample.shape[0], NOISE_DIM).to(device)
            fake_states = G(fake_noise)
            logits_fake = D(fake_states)
            
            D_loss = discriminator_loss(logits_real, logits_fake)
            D_loss.backward()
            D_solver.step()
            
            G_solver.zero_grad()
            fake_noise = sample_noise(sample.shape[0], NOISE_DIM).to(device)
            fake_states = G(fake_noise)
            logits_fake = D(fake_states)
            G_loss = generator_loss(logits_fake)
            G_loss.backward()
            G_solver.step()
        
        with torch.no_grad():
            noise = sample_noise(16, NOISE_DIM).to(device)
            states = G(noise).cpu().numpy()
            visualize_states(env, states, writer, 4, suffix, epoch, render_size = 256)
            


























