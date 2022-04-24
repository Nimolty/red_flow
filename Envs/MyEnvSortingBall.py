import pybullet as p
import time
import pybullet_data
from ipdb import set_trace
import os
import numpy as np
import cv2
from tqdm import tqdm, trange
import time
import pickle
import os

from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyEnv:
    def __init__(self, n_boxes, time_freq=240, is_gui=False):
        if is_gui:
            _ = p.connect(p.GUI)
        else:
            _ = p.connect(p.DIRECT)
        
        my_data_path = './Assets'
        p.setAdditionalSearchPath(my_data_path)
        
        # first set a bse plane
        self.plane_base = p.loadURDF('plane.urdf', [0,0,0], p.getQuaternionFromEuler([0,0,0]))
        self.n_boxes_per_class = n_boxes
        
        # set gravity
        p.setGravity(0, 0, -10)
        
        # set time step
        p.setTimeStep(1. / time_freq)
        self.time_freq = time_freq
        
        ori1 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori2 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori3 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        ori4 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        bound = 0.3
        pos1 = [bound, 0, 0]
        pos2 = [-bound, 0, 0]
        pos3 = [0, bound, 0]
        pos4 = [0, -bound, 0]
        self.bound = bound
        self.r = 0.025
        plane_name = 'plane_transparent.urdf'
        scale = 0.12
        self.transPlane1 = p.loadURDF(plane_name, pos1, ori1, globalScaling=scale)
        self.transPlane2 = p.loadURDF(plane_name, pos2, ori2, globalScaling=scale)
        self.transPlane3 = p.loadURDF(plane_name, pos3, ori3, globalScaling=scale)
        self.transPlane4 = p.loadURDF(plane_name, pos4, ori4, globalScaling=scale)
        
        self.red_balls_1 = []
        self.red_balls_2 = []
        self.red_balls_3 = []
        self.name_mapping_urdf = {'red_1' : 'sphere_red.urdf', 'red_2' : 'sphere_red.urdf', \
                                  'red_3' : 'sphere_red.urdf'}
        self.reset_balls()
        
        # init debug line list
        self.lines = []
        self.n_boxes = self.n_boxes_per_class * 3
        if is_gui:
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
    
    def add_balls(self, center, num, category):
        if category == 'red_1':
            cur_list = self.red_balls_1
        elif category == 'red_2':
            cur_list = self.red_balls_2
        else:
            cur_list = self.red_balls_3
        
        # start init new pos
        flag_load = (len(cur_list) == 0)
        cur_urdf = self.name_mapping_urdf[category]
        scale = 0.05
        iter_list = range(num) if flag_load else cur_list
        # 这里cur_list到底长什么样，一会看一下
        for item in iter_list:
            horizon_p = np.random.normal(size=2) * scale
            horizon_p += center
            horizon_p = np.clip(horizon_p, -(self.bound-self.r), (self.bound-self.r))
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            if flag_load:
                cur_list.append(p.loadURDF(cur_urdf, [horizon_p[0].item(), \
                        horizon_p[1].item(), self.r], cur_ori))
            else:
                p.resetBasePositionAndOrientation(item, [horizon_p[0].item(), \
                        horizon_p[1].item(), self.r], cur_ori)
                p.resetBaseVelocity(item, [0,0,0],[0,0,0])
    
    @staticmethod
    def get_centers(random_flip = True, random_rotate = True):
        # init sample-center for each class
        if random_rotate:
            theta_red_1 = np.random.randint(3) * (2 * np.pi) / 3
        else:
            theta_red_1 = 0
        delta = [-2 * np.pi / 3, 2 * np.pi / 3]
        if random_flip:
            coin = np.random.randint(2)
        else:
            coin = 0
        theta_red_2 = theta_red_1 + delta[coin]
        theta_red_3 = theta_red_1 + delta[1-coin]
        radius = 0.18
        red_1_center = radius * np.array([np.cos(theta_red_1), np.sin(theta_red_1)])
        red_2_center = radius * np.array([np.cos(theta_red_2), np.sin(theta_red_2)])
        red_3_center = radius * np.array([np.cos(theta_red_3), np.sin(theta_red_3)])
        return red_1_center, red_2_center, red_3_center
    
    def pdf_color(self, center, balls, unnorm):
        scale = 0.05
        res = 1
        for i in range(len(balls) // 2):
            ball = balls[i * 2 : (i + 1) * 2]
            if unnorm:
                ball = (self.bound - self.r) * ball
            res *= norm.pdf((ball - center) / scale)
        return res
    
    def pdf(self, state, unnorm=True):
        res = 0
        red_1_center, red_2_center, red_3_center = self.get_centers(False, False)
        red_balls_1, red_balls_2, red_balls_3 = state[:, self.n_boxes_per_class * 2], \
            state[self.n_boxes_per_class * 2:self.n_boxes_per_class * 4], \
                state[self.n_boxes_per_class * 4 : ]
        res = self.pdf_color(red_1_center, red_balls_1, unnorm) * \
            self.pdf_color(red_2_center, red_balls_2, unnorm) * \
                self.pdf_color(red_3_center, red_balls_3, unnorm)
        return res.prod()
    
    def nll(self, state, unnorm=True):
        likelihood = self.pdf(state, unnorm)
        nll = -np.log(likelihood)
        return nll
        
    def reset(self, random=False, random_flip=True, random_rotate=True):
        return self.reset_balls(random, random_flip, random_rotate)
    
    def reset_balls(self, random=False, random_flip=True, random_rotate=True):
        if random:
            # random init pos
            scale = 0.2
            for box in (self.red_balls_1 + self.red_balls_2 + self.red_balls_3):
                horizon_p = np.random.normal(size=2) * scale
                horizon_p = np.clip(horizon_p, -self.bound + self.r, self.bound - self.r)
                cur_ori = p.getQuaternionFromEuler([0, 0, 0])
                p.resetBasePositionAndOrientation(box, [horizon_p[0].item(), horizon[1],item(), self.r], cur_ori)
        else:
            # init to target distribution
            red_1_center, red_2_center, red_3_center = self.get_centers(random_flip, random_rotate)
            red_1_num = self.n_boxes_per_class
            red_2_num = self.n_boxes_per_class
            red_3_num = self.n_boxes_per_class
            self.add_balls(red_1_center, red_1_num, 'red_1')
            self.add_balls(red_2_center, red_2_num, 'red_2')
            self.add_balls(red_3_center, red_3_num, 'red_3')
        # cold start, to prevent the 'floating boxes'
        # for idx in range(20):
        #     p.stepSimulation()
        # 需要先simulation一下，否则无法detect collision
        p.stepSimulation()
        print(self.get_collision_num())
        total_steps = 0
        while self.get_collision_num() > 0:
            for _ in range(20):
                p.stepSimulation()
            total_steps += 20
            if total_steps > 10000:
                print('Warning! Reset takes too much trial!')
                break
        
        self.cur_steps = 0
        self.init_state = self.get_state()
        print(self.get_collision_num())
        return self.init_state
    
    def set_state(self, state, simulation=False, unnorm=True):
        assert state.shape[0] == len(self.red_balls_1 + self.red_balls_2 + self.red_balls_3) * 2
        for idx, boxID in enumerate(self.red_balls_1 + self.red_balls_2 + self.red_balls_3):
            cur_state = state[idx * 2 : (idx + 1) * 2]
            if unnorm:
                cur_pos = (self.bound - self.r) * cur_state
            else:
                cur_pos = cur_state
            cur_ori = p.getQuaternionFromEuler([0,0,0])
            p.resetBasePositionAndOrientation(boxID, [cur_pos[0], cur_pos[1], 0], cur_ori)
            p.resetBaseVelocity(boxId, [0,0,0], [0,0,0])
        if simulation:
            for idx in range(20):
                p.stepSimulation()
    
    def count_state_collision(self, state, unnorm=True, thickness=-5e-2):
        assert state.shape[0] == len(self.red_balls_1 + self.red_balls_2 + self.red_balls_3) * 2
        if unnorm:
            state = state * (self.bound - self.r)
        state = state.reshape(-1, 1, 2)
        dist = np.linalg.norm(state - state.reshape(1, -1, 2), axis = -1)
        dist -= 2 * (1 + thickness) * self.r
        dist += np.eye(dist.shape[0]) * 1e4
        collision = dist < 0
        collision_num = collision.sum() // 2
        collision_score = (dist * collision).sum() // 2
        
        return collision_num, collision_score
    
    def get_collision_num(self, centralized=True):
        items = self.red_balls_1 + self.red_balls_2 + self.red_balls_3
        collisions = np.zeros((len(items), len(items)))
        for idx1, ball_1 in enumerate(items[:-1]):
            for idx2, ball_2 in enumerate(items[idx1+1:]):
                points = p.getContactPoints(ball_1, ball_2)
                collisions[idx1][idx2] = (len(points) > 0)
        return np.sum(collisions).item() if centralized else collisions
    
    def set_volicity(self, velocity, simulation_steps = 20):
        for idx, boxId in enumerate(self.red_balls_1 + self.red_balls_2 + self.red_balls_3):
            cur_velocity = velocity[idx * 2 : (idx + 1) * 2]
            # 质心的速度和自转角速度= 0
            p.resetBaseVelocity(boxId, [cur_velocity[0], cur_velocity[1], 0], [0,0,0])
        for idx in enumerate(simulation_steps):
            p.stepSimulation()
    
    def check_valid(self):
        positions = []
        for ballId in (self.red_balls_1 + self.red_balls_2 + self.red_balls_3):
            pos, ori = p.getBasePositionAndOrientation(ballId)
            positions.append(pos)
        positions = np.stack(positions)

        # 所有物体都得在界内
        flag_x_bound = np.max(positions[:, 0:1]) <= (self.bound-self.r) and np.min(positions[:, 0:1]) >= (-self.bound+self.r)
        flag_y_bound = np.max(positions[:, 1:2]) <= (self.bound-self.r) and np.min(positions[:, 1:2]) >= (-self.bound+self.r)

        # 得贴在平面上，重心得和半径一致
        flag_height = np.max(np.abs(positions[:, -1:] - self.r)) < 0.001
        return flag_height&flag_x_bound&flag_y_bound, (positions[:, -1:])

    def get_state(self, norm=True):
        # note that return is normalized 
        # 注意一定要严格按照123顺序来
        
        box_states = []
        for boxId in (self.red_balls_1 + self.red_balls_2 + self.red_balls_3):
            pos, ori = p.getBasePositionAndOrientation(boxId)
            
            pos = np.array(pos[0:2], dtype = np.float32)
            
            if norm:
                pos = pos / (self.bound - self.r)
            box_state = pos
            box_states.append(box_state)
        box_states = np.concatenate(box_states, axis=0)
        assert box_states.shape[0] == len(self.red_balls_1 + self.red_balls_2 + self.red_balls_3) * 2
        return box_states
    
    def set_brownian_velocity(self):
        scale = 0.1
        for boxId in self.boxes:
            horizon_v = np.random.normal(size=2)*scale
            vel = [horizon_v[0].item(), horizion_v[1].item(), 0]
            p.resetBaseVelocity(boxId, linearVelocity=vel)
    
    def add_lines(self, score):
        state = self.get_state(False)
        norm_factor = np.linalg.norm(np.reshape(score, (self.n_boxes, 2)), axis = -1).max()
        score = score * 0.1 / norm_factor
        origins = [state[idx * 2:(idx + 1) * 2] for idx in range(self.n_boxes)]
        deltas = [score[idx * 2:(idx + 1) * 2] for idx in range(self.n_boxes)]
        targets = [origin + delta for origin, delta in zip(origins, deltas)]
        for origin, target in zip(origins, targets):
            self.lines.append(p.addUserDebugLine(
                    [origin[0], origin[1], self.r],
                    [target[0], target[1], self.r],
                    lifeTime=0,
                    lineWidth=2,
                    lineColorRGB=[139, 0, 0],
                )
            )
            
    
    @staticmethod
    def remove_Lines():
        p.removeAllUserDebugItems()
    
    def render(self, img_size, score=None):
        #if score is not None:
        #    self.add_lines(score)
        viewmatrix = p.computeViewMatrix(
            cameraEyePosition = [0, 0.0, 1.0],
            cameraTargetPosition = [0,0,0],
            cameraUpVector=[0,1,0],
            )
        projectionmatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1,
        )
        _, _, rgba, _, _ = p.getCameraImage(img_size, img_size, viewMatrix=viewmatrix,
                                            projectionMatrix=projectionmatrix)
        rgb = rgba[:, :, 0:3]
        
        return rgb
    @staticmethod
    def step():
        p.stepSimulation()
    @staticmethod
    def close():
        p.disconnect()

#nimol = MyEnv(5, time_freq = 240, is_gui = True)
#nimol.close()
        
        
        
        
        
        
        