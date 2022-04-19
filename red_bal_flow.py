# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:27:24 2022

@author: lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:44:48 2022

@author: lenovo
"""
import pickle
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import sys
from nflows.flows.base import Flow
from nflows.distributions.uniform import BoxUniform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform 
from nflows.transforms.permutations import ReversePermutation 
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_sample(num, model):
    return model[0:num]

def save_imgs_to_video(eval_states, save_path, output_num, suffix = ''):
    fps = 50
    render_size = 256
    imgs = []
    for idx, box_state in enumerate(eval_states):
        env.set_state(box_state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis = 0)
    images_to_video(save_path + '.avi', batch_imgs, fps, (render_size, render_size))

def line_sample(output_num, num_boxes, flow, start, end):
    x = torch.linspace(start, end, output_num).unsqueeze(1).to(device)
    #y = torch.linspace(start, end, output_num).unsqueeze(1).to(device)
    latent_z = torch.concat([x] * 2 * num_boxes, dim = 1).to(device)
    return flow._transform.inverse(latent_z)[0].detach().cpu().numpy()
    
def _log_prob(flow, inputs):
    #embedded_context = self._embedding_net(context)
    noise, logabsdet = flow._transform(inputs)
    noise = noise.to(device)
    logabsdet = logabsdet.to(device)
    log_prob = flow._distribution.log_prob(noise).to(device)
    return log_prob + logabsdet 

def visual_states(env, states, nrow, suffix, render_size = 256):
    imgs = []
    for state in states:
        env.set_state(state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    return grid




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type = str)
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
    
    with open(f'./dataset/exp_0415_name_1.pth', 'rb') as f:
        samples = pickle.load(f)
    
    #samples = select_sample(sample_num, model) / (wall_bound - 0.025)
    num_layers = 5
    #base_dist = BoxUniform(torch.tensor([-1.0, -1.0]*num_boxes), torch.tensor([1.0, 1.0] * num_boxes))
    base_dist = StandardNormal(shape = [2 * num_boxes])
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
    loader = DataLoader(TensorDataset(samples, targets), batch_size = 2048)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2 * num_boxes))
        transforms.append(MaskedAffineAutoregressiveTransform(features = 2 * num_boxes, hidden_features= 4 * num_boxes))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist).to(device)
    optimizer = optim.Adam(flow.parameters())

    for epoch in trange(epoches):
        #st = time.time()
        for i, sample in enumerate(loader):
            sample = sample[0].to(device)
            optimizer.zero_grad()
            loss = -_log_prob(flow, sample).mean()
            loss.backward()
            optimizer.step()
        # print('loss', loss.data)
            writer.add_scalar('Loss', loss, i + epoch * len(loader))
            #noise_z = flow._distribution.rsample(torch.Size([16])).to(device)
            #samples, _ = flow._transform.inverse(noise_z)
        #with torch.no_grad():
        states = flow.sample(16).to(device) 
        states = states.clone().to(torch.float64)       #print(states)
        visualize_states(env, states, writer, 4, suffix, epoch, render_size = 256)
        #print(-st + time.time())
        time.sleep(0.003)
    
    #eval_states = line_sample(output_num, num_boxes, flow, -1.5, 1.5)
    #save_imgs_to_video(eval_states, save_path, output_num, suffix = '')
    env.close()
    writer.close()
    































