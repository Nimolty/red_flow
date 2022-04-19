import time
import pickle
import random
from ipdb import set_trace

from gym import spaces
import numpy as np
import pybullet as p


# import pybullet_data

class Sorting:
    def __init__(self, max_episode_len=500, **kwargs):
        n_boxes = kwargs['n_boxes']
        wall_bound = kwargs['wall_bound']
        time_freq = kwargs['time_freq']
        is_gui = kwargs['is_gui']

        if is_gui:
            self.cid = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self.cid = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
        my_data_path = './Assets'
        p.setAdditionalSearchPath(my_data_path)  # optionally

        # first set a base plane
        self.plane_base = p.loadURDF("plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                                     physicsClientId=self.cid)
        self.n_boxes = n_boxes

        # set gravity
        p.setGravity(0, 0, -10, physicsClientId=self.cid)

        # set time step
        p.setTimeStep(1. / time_freq, physicsClientId=self.cid)
        self.time_freq = time_freq

        # then set 4 transparent planes surrounded
        ori1 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori2 = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ori3 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        ori4 = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        pos1 = [wall_bound, 0, 0]
        pos2 = [-wall_bound, 0, 0]
        pos3 = [0, wall_bound, 0]
        pos4 = [0, -wall_bound, 0]
        self.bound = wall_bound
        self.r = 0.025
        plane_name = "plane_transparent.urdf"
        scale = wall_bound / 2.5
        self.transPlane1 = p.loadURDF(plane_name, pos1, ori1, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane2 = p.loadURDF(plane_name, pos2, ori2, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane3 = p.loadURDF(plane_name, pos3, ori3, globalScaling=scale, physicsClientId=self.cid)
        self.transPlane4 = p.loadURDF(plane_name, pos4, ori4, globalScaling=scale, physicsClientId=self.cid)

        # init ball list for R,G,B balls
        self.blue_balls = []
        self.name_mapping_urdf = {'blue': "sphere_blue.urdf"}


        self.max_episode_len = max_episode_len
        self.num_episodes = 0

        # reset cam-pose
        if is_gui:
            # reset cam-pose to a top-down view
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0., cameraPitch=-89.,
                                         cameraTargetPosition=[0, 0, 0], physicsClientId=self.cid)

    @staticmethod
    def seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def add_balls(self, positions, category):
        """
        load balls at given positions
        category in ['blue']
        positions: [n_boxes, 2] # 2-d coordinates
        """
        cur_list = self.blue_balls
        flag_load = (len(cur_list) == 0)
        cur_urdf = self.name_mapping_urdf[category]
        iter_list = range(self.n_boxes) if flag_load else cur_list
        radius_list = [1.0, 1.5, 2.0]
        for i, item in enumerate(iter_list):
            horizon_p = positions[i]
            horizon_p = np.clip(horizon_p, -(self.bound - self.r), (self.bound - self.r))
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            if flag_load:
                cur_list.append(p.loadURDF(cur_urdf, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori,
                                           physicsClientId=self.cid))
            else:
                p.resetBasePositionAndOrientation(item, [horizon_p[0].item(), horizon_p[1].item(), self.r], cur_ori,
                                                  physicsClientId=self.cid)
                p.resetBaseVelocity(item, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def get_positions(self):
        """
        sample i.i.d. gaussian 2-d positions
        return positions: [n_boxes, 2]
        """
        scale = (self.bound - self.r) / 1
        positions = np.random.uniform(-1, 1, size=(self.n_boxes, 2)) * scale
        return positions


    def set_state(self, state, verbose=None):
        """
        set 2-d positions for each object
        state: [n_balls, 2]
        """
        assert state.shape[0] == len(self.blue_balls) * 2
        for idx, boxId in enumerate(self.blue_balls):
            cur_state = state[idx * 2:(idx + 1) * 2]
            # un-normalize
            cur_pos = (self.bound - self.r) * cur_state
            # cur_pos = 1.0 * cur_state
            cur_ori = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(boxId, [cur_pos[0], cur_pos[1], 0], cur_ori, physicsClientId=self.cid)
            p.resetBaseVelocity(boxId, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def check_valid(self):
        """
        check whether all objects are physically correct(no floating balls)
        """
        positions = []
        for ballId in self.blue_balls:
            pos, ori = p.getBasePositionAndOrientation(ballId, physicsClientId=self.cid)
            positions.append(pos)
        positions = np.stack(positions)

        # 所有物体都得在界内
        flag_x_bound = np.max(positions[:, 0:1]) <= (self.bound - self.r) and np.min(positions[:, 0:1]) >= (
                    -self.bound + self.r)
        flag_y_bound = np.max(positions[:, 1:2]) <= (self.bound - self.r) and np.min(positions[:, 1:2]) >= (
                    -self.bound + self.r)

        # 得贴在平面上，重心得和半径一致
        flag_height = np.max(np.abs(positions[:, -1:] - self.r)) < 0.001
        return flag_height & flag_x_bound & flag_y_bound, (positions[:, -1:])

    def get_state(self):
        """
        get 2-d positions for each object
        return: [n_balls*2]
        """
        # 注意一定要严格按照rgb 顺序来！
        box_states = []
        for boxId in self.blue_balls:
            pos, ori = p.getBasePositionAndOrientation(boxId, physicsClientId=self.cid)

            pos = np.array(pos[0:2], dtype=np.float32)
            # normalize -> [-1, 1]
            pos = pos / (self.bound - self.r)

            box_state = pos
            box_states.append(box_state)
        box_states = np.concatenate(box_states, axis=0)
        assert box_states.shape[0] == len(self.blue_balls) * 2
        return box_states

    def set_velocity(self, vels):
        """
        set 2-d linear velocity for each object
        vels: [n_balls, 2]
        """
        # vels.shape = [num_boxes, 2]
        # set_trace()
        for boxId, vel in zip(self.blue_balls, vels):
            vel = [vel[0].item(), vel[1].item(), 0]
            # print(vel)
            # set_trace()
            p.resetBaseVelocity(boxId, linearVelocity=vel, physicsClientId=self.cid)

    def render(self, img_size):
        """
        return an  image of  cur state: [img_size, img_size, 3], BGR
        """
        # if grad exists, then add debug line
        viewmatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0.0, 1.0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )
        print('view ok')
        projectionmatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1,
        )
        print('project ok')
        _, _, rgba, _, _ = p.getCameraImage(img_size, img_size, viewMatrix=viewmatrix,
                                            projectionMatrix=projectionmatrix, physicsClientId=self.cid)
        print('rgb ok')
        rgb = rgba[:, :, 0:3]

        return rgb


    def get_collision_num(self):
        """
        return the collision number at current step
        collision_num = sum_{1 <= i < j <= K} is_collision(object_i, object_j)
        """
        # collision detection
        items = self.blue_balls
        cnt = 0
        for idx1, ball_1 in enumerate(items[:-1]):
            for idx2, ball_2 in enumerate(items[idx1 + 1:]):
                points = p.getContactPoints(ball_1, ball_2, physicsClientId=self.cid)
                cnt += (len(points) > 0)
                # for debug
                # print(f'{name1} {name2} {len(points)}')
        return cnt

    @staticmethod
    def sample_action():
        raise NotImplementedError

    def reset(self, is_random=False):
        raise NotImplementedError

    def step(self, vels, duration=10):
        raise NotImplementedError

    def close(self):
        p.disconnect(physicsClientId=self.cid)


class RLSorting(Sorting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 这里得用gym-env的spaces，而不是tf-agents的spaces
        self.max_action = kwargs['max_action']
        self.action_space = spaces.Box(-self.max_action, self.max_action, shape=(2 * self.n_boxes,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(2 * self.n_boxes,), dtype=np.float32)
        self.cur_steps = 0


    def nop(self):
        """
        no operation at this time step
        """
        action = np.zeros((self.n_boxes, 2))
        return self.step(action)


    def step(self, vels, step_size=8):
        """
        vels: [n_balls, 2], 2-d linear velocity
        for each dim,  [-max_vel,  max_vel]
        """
        # action: numpy, [num_box*2]
        collision_num = 0
        old_pos = self.get_state().reshape(self.n_boxes, 2)

        vels = np.reshape(vels, (len(self.blue_balls), 2))
        # max_vel_norm = np.max((np.max(np.linalg.norm(vels, ord=np.inf, axis=-1)), 1e-7))
        max_vel_norm = np.max(np.abs(vels))
        scale_factor = self.max_action / max_vel_norm
        scale_factor = np.min([scale_factor, 1])
        vels = scale_factor * vels
        # vels = np.clip(vels, -self.max_action, self.max_action) # clip to action spaces
        max_vel = vels.max()
        if max_vel > self.max_action:
            print(f'!!!!!current max velocity {max_vel} exceeds max action {self.max_action}!!!!!')
        self.set_velocity(vels)
        # set_trace()
        # old_state = self.get_state()
        for _ in range(step_size):
            p.stepSimulation(physicsClientId=self.cid)
            self.set_velocity(vels)
            collision_num += self.get_collision_num()
        # new_state = self.get_state()
        collision_num /= step_size # 因为有可能碰撞持续了很久，你要是每个step都算就太多了，算个平均就好了
        # judge if is done
        self.cur_steps += 1
        is_done = self.cur_steps >= self.max_episode_len
        # print(f'{self.cur_steps}, {is_done}, time cost: {time.time() - t_s}')

        new_pos = self.get_state().reshape(self.n_boxes, 2)
        delta_pos = new_pos - old_pos
        vel_err = np.max(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        vel_err_mean = np.mean(np.abs(delta_pos*self.time_freq*self.bound/step_size - vels))/self.max_action
        # 其实下面这个，如果发生严重碰撞，那也会产生大量warnings
        # if vel_err_mean > 0.1:
        #     print(f'Warning! Large mean-vel-err at cur step: {vel_err}!')

        return self.get_state(), is_done, {'delta_pos': delta_pos, 'collision_num': collision_num,
                                              'vel_err': vel_err, 'vel_err_mean': vel_err_mean}


    def reset(self):
        self.num_episodes += 1
       # t_s = time.time()

        blue_positions = self.get_positions()

        self.add_balls(blue_positions, 'blue')

        # 巨坑！你要是不先simulate一下，它没法detect collision啊！
        p.stepSimulation(physicsClientId=self.cid)
        total_step = 0
        while self.get_collision_num() > 0:
            for _ in range(20):
                p.stepSimulation(physicsClientId=self.cid)
            total_step += 20
            if total_step > 1000:
                break
        # set_trace()
        self.cur_steps = 0
       # print(f'No.{self.num_episodes} Episodes, reset now! time cost: {time.time() - t_s}')
        return self.get_state()

