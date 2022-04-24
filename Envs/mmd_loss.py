# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:21:45 2022

@author: lenovo
"""
import torch

def gaussian_kernel(source, target):
    n_samples = int(source.shape[0]) + int(source.shape[0])
    total = torch.concat((source, target), axis = 0)
    total0 = torch.unsqueeze(total, dim = 0)
    total0 = torch.broadcast_to(total0, [total.shape[0], total.shape[0], total.shape[1]])
    
    total1 = torch.unsqueeze(total, dim = 1)
    total1 = torch.broadcast_to(total1, [total.shape[0], total.shape[0], total.shape[1]])
    
    L2_distance_square = torch.sum(torch.square(total1 - total0), dim=2)
    bandwidth = torch.sum(L2_distance_square) / (n_samples**2 - n_samples)
    kernel_val = torch.exp(-L2_distance_square / bandwidth)
    return kernel_val

def inverse_multiquadric(source, target):
    n_samples = int(source.shape[0]) + int(source.shape[0])
    total = torch.concat((source, target), axis = 0)
    total0 = torch.unsqueeze(total, dim = 0)
    total0 = torch.broadcast_to(total0, [total.shape[0], total.shape[0], total.shape[1]])
    
    total1 = torch.unsqueeze(total, dim = 1)
    total1 = torch.broadcast_to(total1, [total.shape[0], total.shape[0], total.shape[1]])
    
    L2_distance_square = torch.sum(torch.square(total1 - total0), dim=2)
    bandwidth = torch.sum(L2_distance_square) / (n_samples**2 - n_samples)
    kernel_val = 1 / (1 + L2_distance_square / bandwith)
    return kernel_val

def mmd(source, target):
    batch_size = source.shape[0]
    kernel = gaussian_kernel(source, target)
    XX = kernel[:batch_size, :batch_size]
    YY = kernel[batch_size:, batch_size:]
    XY = kernel[:batch_size, batch_size:]
    YX = kernel[batch_size:, :batch_size]
    
    loss = torch.mean(XX+YY-XY-YX)
    return loss



























