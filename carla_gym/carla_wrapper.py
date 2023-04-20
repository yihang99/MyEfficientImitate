import cv2
from collections import deque

import gym
from gym import spaces
import ipdb
import numpy as np


class CarlaWrapper(gym.Wrapper):
    """
    A clean wrapper of carla environment, with continuous action space.
    """
    def __init__(self, env,
                 cvt_string=True,
                 test=False,
                 setting='default',
                 frame_skip=1):
        """
        :param env: instance of gym environment
        """
        super(CarlaWrapper, self).__init__(env)
        self.cvt_string = cvt_string
        self.test = test
        self.setting = setting
        self.frame_skip = frame_skip

        # self.obs_shape = 50, 300
        self.obs_shape = 160, 960

        if self.setting == 'pure_rgb':
            shape = *self.obs_shape, 3
            low = np.zeros(shape=shape, dtype='uint8')
            high = low + 255
            self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype='uint8')
        self.action_space = spaces.Box(low=np.array([-1., -1.]),
                                       high=np.array([1., 1.]),
                                       shape=(2,), dtype='float32')
        self.step_cnt = 0

    def preprocess_rgb_data(self, obs_dict):
        obs_image = np.concatenate((obs_dict['left_rgb']['data'][160:320, 320:640],
                                    obs_dict['central_rgb']['data'][160:320, 320:640],
                                    obs_dict['right_rgb']['data'][160:320, 320:640]), axis=1)
        obs_image = cv2.resize(obs_image, (self.obs_shape[1], self.obs_shape[0]),
                               interpolation=cv2.INTER_AREA).astype(np.uint8)
        # self.obs_shape = 320, 960
        # obs_image = np.concatenate((obs_dict['left_rgb']['data'][80:400, 320:640],
        #                             obs_dict['central_rgb']['data'][80:400, 320:640],
        #                             obs_dict['right_rgb']['data'][80:400, 320:640]), axis=1)
        # obs_image = cv2.resize(obs_image, (self.obs_shape[1], self.obs_shape[0]),
        #                        interpolation=cv2.INTER_AREA).astype(np.uint8)
        return obs_image

    def step(self, action):
        """

        :param action: could be dict or tuple
        :return:
        """
        if type(action) == dict:
            control_dict = action
        elif type(action) == tuple:
            steer, throttle = float(action[0]), float(action[1])
            if throttle > 0.:
                control_dict = {'steer': steer, 'throttle': throttle, 'brake': 0.}
            else:
                control_dict = {'steer': steer, 'throttle': 0, 'brake': - throttle}

        # obs_dict, reward, done, info = self.env.step(control_dict)
        reward = 0
        obs_dict, r, done, info = None, None, None, None
        for k in range(self.frame_skip):
            obs_dict, r, done, info = self.env.step(control_dict)
            reward += r
            if done:
                break

        self.step_cnt += 1

        obs_image = self.preprocess_rgb_data(obs_dict)

        # obs_image = obs_image.astype(np.uint8)
        if self.setting == 'pure_rgb':
            return obs_image, reward, done, info

        elif self.setting == 'default':
            if self.test:
                obs_dict['ori_data'] = obs_image
            if self.cvt_string:
                pass
                # obs_image = arr_to_str(obs_image)
            obs_dict['data'] = obs_image
            if obs_dict['gnss']['command'][0] < 0:
                obs_dict['gnss']['command'][0] = 4
            return obs_dict, reward, done, info

    def reset(self, test=False, **kwargs):
        self.step_cnt = 0
        # if self.test:
        #     self.env.set_task_idx(np.random.choice(self.env.num_tasks))
        obs_dict, info = self.env.reset(**kwargs)

        obs_image = self.preprocess_rgb_data(obs_dict)
        if self.setting == 'pure_rgb':
            return obs_image

        elif self.setting == 'default':
            if self.test:
                obs_dict['ori_data'] = obs_image
            if self.cvt_string:
                pass
                # obs_image = arr_to_str(obs_image)
            obs_dict['data'] = obs_image
            return obs_dict, info

    def close(self):
        self.env.close()

    def set_task_idx(self, task_idx):
        self.env.set_task_idx(task_idx)

    def transfer_obs_to_transfuser_format(self, obs_dict):
        input_data = {}
        input_data['rgb_right'] = 0, cv2.cvtColor(obs_dict['right_rgb']['data'], cv2.COLOR_BGR2RGB)
        input_data['rgb_left'] = 0, cv2.cvtColor(obs_dict['left_rgb']['data'], cv2.COLOR_BGR2RGB)
        input_data['rgb_front'] = 0, cv2.cvtColor(obs_dict['central_rgb']['data'], cv2.COLOR_BGR2RGB)
        # input_data['rgb_right'] = 0, obs_dict['right_rgb']['data']
        # input_data['rgb_left'] = 0, obs_dict['left_rgb']['data']
        # input_data['rgb_front'] = 0, obs_dict['central_rgb']['data']
        input_data['gps'] = 0, obs_dict['gnss']['gnss']
        input_data['imu'] = 0, obs_dict['gnss']['imu']
        input_data['speed'] = 0, {'speed': obs_dict['speed']['speed'].item()}
        return input_data
