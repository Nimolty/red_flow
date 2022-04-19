import os
import cv2
from ipdb.__main__ import set_trace
import numpy as np
import torch
from torch.utils.data import Subset
from PIL import Image

from tqdm import tqdm

from torchvision.utils import make_grid
import random
import matplotlib.pyplot as plt

import pickle
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

from ipdb import set_trace


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def string2bool(str_bool):
    return str_bool == 'True'


def split_dataset(dataset, seed, test_ratio, full_train='False'):
    random.seed(seed)

    # get train and test indices
    # get test split according to mode
    if dataset.mode == 'multi':
        items_dict = dataset.items_dict
        test_num = int(len(items_dict.keys()) * test_ratio)
        test_keys = random.sample(list(items_dict.keys()), test_num)
        test_indics = []
        for key in test_keys:
            test_indics += items_dict[key]
    else:
        test_num = int(len(dataset) * test_ratio)
        test_indics = random.sample(range(len(dataset)), test_num)

    # get train according to test
    train_indics = list(set(range(len(dataset))) - set(test_indics))

    # assertion of indices
    assert len(train_indics) + len(test_indics) == len(dataset)
    assert len(set(train_indics) & set(test_indics)) == 0

    # split dataset according to indices
    test_dataset = Subset(dataset, test_indics)
    train_dataset = dataset if full_train == 'True' else Subset(dataset, train_indics)

    # log infos
    infos_dict = {
        'test_indices': test_indics,
        'train_indices': train_indics,
        'room_num': len(dataset.items_dict.keys()),
    }
    return train_dataset, test_dataset, infos_dict


class GraphDataset:
    def __init__(self, data_name, base_noise_scale=0.01, data_ratio=1):
        self.data_root = f'./dataset/{data_name}/content'
        self.folders_path = os.listdir(self.data_root)
        self.items = []
        self.items_dict = {}
        ptr = 0
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            assert len(files_list) % 2 == 0
            if files not in self.items_dict.keys():
                self.items_dict[files] = []
            for idx in range(int((len(files_list)*data_ratio)//2)):
                item = {
                    'wall_path': cur_folder_path+f'{idx+1}_wall.pickle',
                    'obj_path': cur_folder_path+f'{idx+1}_obj.pickle',
                    'room_name': files,
                }
                self.items.append(item)
                self.items_dict[files].append(ptr)
                ptr += 1
            
        self.state_dim = 4
        self.size_dim = 2
        # self.wall_dim = 6
        self.scale = base_noise_scale
        self.histogram_path = f'./dataset/{data_name}/histogram.png'

        # 检查有没有数据集histogram信息，没有就补一个
        self.draw_histogram()

        # 检查是单个房间还是多个房间
        self.mode = 'multi' if len(self.items_dict.keys()) > 1 else 'single'

    def draw_histogram(self):
        plt.figure(figsize=(10,10))
        histogram = []
        # set_trace()
        for files in self.folders_path:
            cur_folder_path = f'{self.data_root}/{files}/'
            files_list = os.listdir(cur_folder_path)
            # 因为有俩文件！obj-wall
            histogram.append(len(files_list) // 2)
            # print(len(files_list)//2)
        histogram = np.array(histogram)
        plt.hist(histogram, bins=4)
        plt.title(f'Total room num: {len(self.folders_path)}')
        plt.savefig(self.histogram_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        item_path = self.items[item]
        with open(item_path['wall_path'], 'rb') as f:
            wall_feat = pickle.load(f)
        with open(item_path['obj_path'], 'rb') as f:
            obj_batch = pickle.load(f)
        wall_feat = torch.tensor(wall_feat).float()
        obj_batch = torch.tensor(obj_batch)

        # edge_wall = knn_graph(wall_feat, wall_feat.shape[0]-1)
        # 只考虑position算nearest neighbor，一半物体
        # edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0]//2+1)
        edge_obj = knn_graph(obj_batch[:, 0:2], obj_batch.shape[0]-1) # fully connected
        # data_wall = Data(x=wall_batch.float(),
        #                  edge_index=edge_wall)
        data_obj = Data(x=obj_batch[:, 0:self.state_dim].float(),
                        geo=obj_batch[:, self.state_dim:self.state_dim+self.size_dim].float(),
                        category=obj_batch[:, -1:].long(),
                        edge_index=edge_obj)
        # augment the data with slight perturbation
        scale = self.scale
        # data_wall.x += torch.randn_like(data_wall.x) * scale
        # only pos, not ori
        data_obj.x += torch.cat([torch.randn_like(data_obj.x[:, 0:2]), torch.zeros_like(data_obj.x[:, 2:4])], dim=1) * scale
        # data_obj.geo += torch.randn_like(data_obj.geo) * scale
        # wall_feat += torch.randn_like(wall_feat) * scale
        return wall_feat, data_obj, item_path['room_name']

# def images_to_video(path, images, fps, size):
#     out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#     for item in images:
#         out.write(item)
#     out.release()

def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for item in images:
        out.write(item)
    out.release()


def save_video_cond(sampler, ref_batch, samples, save_path, fps = 50, render_size = 256, render_height=10.):
    wall_batch, obj_batch, names = ref_batch
    ptr = obj_batch.ptr
    for idx in range(ptr.shape[0]-1):
        cur_states = samples[:, ptr[idx]: ptr[idx+1], :].to(obj_batch.x.device)
        sim = sampler[names[idx]]
        sim.normalize_room()
        # sim.clear_body()
        # generated
        imgs = []
        for cur_state in tqdm(cur_states, desc=f'Saving video for [{idx+1}/{ptr.shape[0]-1}] video'):
            # set_trace()
            cur_state = torch.cat(
                [cur_state,
                 obj_batch.geo[ptr[idx]: ptr[idx+1]],
                 obj_batch.category[ptr[idx]: ptr[idx+1]]], dim=-1)
            sim.set_state(cur_state.cpu().numpy(), wall_batch[idx].cpu().numpy())
            img = sim.take_snapshot(render_size, height=render_height)
            # BGR -> RGB
            imgs.append(img[...,::-1])
        # close sim
        sim.disconnect()
        batch_imgs = np.stack(imgs, axis=0)
        # images_to_video(save_path+f'_No.{idx+1}.avi', batch_imgs, fps, (render_size, render_size))

        # save mp4 videos using cv2
        size = (render_size, render_size)
        out = cv2.VideoWriter(save_path+f'_No.{idx+1}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(batch_imgs)):
            # rgb_img = cv2.cvtColor(img_array[i], cv2.COLOR_RGB2BGR)
            rgb_img = batch_imgs[i]
            out.write(rgb_img)
        out.release()


def save_gif_cond(sampler, ref_batch, samples, save_path, fps = 50, render_size = 256, render_height=10.):
    wall_batch, obj_batch, names = ref_batch
    ptr = obj_batch.ptr
    for idx in range(ptr.shape[0]-1):
        cur_states = samples[:, ptr[idx]: ptr[idx+1], :].to(obj_batch.x.device)
        sim = sampler[names[idx]]
        sim.normalize_room()
        # sim.clear_body()
        # generated
        imgs = []
        for cur_state in tqdm(cur_states, desc=f'Saving video for [{idx+1}/{ptr.shape[0]-1}] video'):
            # set_trace()
            cur_state = torch.cat(
                [cur_state,
                 obj_batch.geo[ptr[idx]: ptr[idx+1]],
                 obj_batch.category[ptr[idx]: ptr[idx+1]]], dim=-1)
            sim.set_state(cur_state.cpu().numpy(), wall_batch[idx].cpu().numpy())
            img = sim.take_snapshot(render_size, height=render_height)
            imgs.append(img)
        # close sim
        sim.disconnect()
        imgs = [Image.fromarray(img) for img in imgs]
        # duration: ms
        imgs[0].save(save_path + '.gif', save_all=True, append_images=imgs[1:], loop=0)


def save_video(env, states, save_path, simulation=False, fps = 50, render_size = 256, velocity=False):
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving video'):
        if velocity:
            env.set_velocity(state, simulation)
        else:
            env.set_state(state, simulation)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    images_to_video(save_path+'.avi', batch_imgs, fps, (render_size, render_size))
    # 完事了要reset下
    env.reset_balls(True)


def save_gif(env, states, save_path, simulation=False, fps = 50, render_size = 256, velocity=False):
    imgs = []
    for _, state in tqdm(enumerate(states), desc='Saving gif'):
        if velocity:
            env.set_velocity(state, simulation)
        else:
            env.set_state(state, simulation)
        img = env.render(render_size)
        imgs.append(img)
    imgs = [Image.fromarray(img) for img in imgs]
    # duration: ms
    imgs[0].save(save_path+'.gif', save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)

    # 完事了要reset下
    env.reset_balls(True)


def visualize_states(env, states, logger, nrow, suffix, epoch, render_size = 256):
    imgs = []
    for state in states:
        env.set_state(state)
        img = env.render(render_size)
        imgs.append(img)
    batch_imgs = np.stack(imgs, axis=0)
    ts_imgs = torch.tensor(batch_imgs).permute(0, 3, 1, 2)
    grid = make_grid(ts_imgs.float(), padding=2, nrow=nrow, normalize=True)
    logger.add_image(f'Images/dynamic_{suffix}', grid, epoch)


class ReplayBufferCond(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
