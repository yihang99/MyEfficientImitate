import cv2
from collections import deque

import ipdb
import numpy as np
from core.game import Game
from core.utils import arr_to_str


class CarlaWrapper(Game):
    def __init__(self, env, discount: float, num_steers=9, num_throts=3, cvt_string=True, test=False):
        """
        :param env: instance of gym environment
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string
        self.test = test
        self.three_cam = env.three_cam

        # Action bins
        self.num_steers = num_steers
        self.num_throts = num_throts
        _steers = np.linspace(-1.0, 1.0, self.num_steers)
        _throts = np.linspace(0, 1.0, self.num_throts)

        _steers = np.append(np.tile(_steers, self.num_throts), 0.)
        _throts = np.append(np.repeat(_throts, self.num_steers), 0.)
        _brakes = np.append(np.zeros(len(_steers) - 1), 1.0)

        self._actions = np.stack([_steers, _throts, _brakes], axis=-1)

        self.obs_shape = (96, 96) if not self.three_cam else (240, 480)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        steer, throttle, brake = self._actions[action]
        control_dict = {'steer': steer, 'throttle': throttle, 'brake': brake}
        obs_dict, reward, done, info = self.env.step(control_dict)
        if self.three_cam:
            obs_image = np.concatenate((obs_dict['left_rgb']['data'][160:320, 320:640],
                                        obs_dict['central_rgb']['data'][160:320, 320:640],
                                        obs_dict['right_rgb']['data'][160:320, 320:640]), axis=1)
        else:
            obs_image = obs_dict['central_rgb']['data']

        obs_image = obs_image.astype(np.uint8)
        # ToDo: add seg
        if self.test:
            obs_dict['ori_data'] = obs_image

        # obs_image = cv2.resize(obs_image, (self.obs_shape[1], self.obs_shape[0]),
        #                        interpolation=cv2.INTER_AREA).astype(np.uint8)

        if self.cvt_string:
            obs_image = arr_to_str(obs_image)
        obs_dict['data'] = obs_image

        if obs_dict['gnss']['command'][0] < 0:
            obs_dict['gnss']['command'][0] = 4

        return obs_dict, reward, done, info

    def reset(self, test=False, **kwargs):
        # if self.test:
        #     self.env.set_task_idx(np.random.choice(self.env.num_tasks))
        obs_dict, info = self.env.reset(**kwargs)
        if self.three_cam:
            obs_image = np.concatenate((obs_dict['left_rgb']['data'], obs_dict['central_rgb']['data'],
                                        obs_dict['right_rgb']['data']), axis=1)
        else:
            obs_image = obs_dict['central_rgb']['data']

        obs_image = obs_image.astype(np.uint8)

        if self.test:
            obs_dict['ori_data'] = obs_image

        obs_image = cv2.resize(obs_image, (self.obs_shape[1], self.obs_shape[0]),
                               interpolation=cv2.INTER_AREA).astype(np.uint8)

        if self.cvt_string:
            obs_image = arr_to_str(obs_image)

        obs_dict['data'] = obs_image

        return obs_dict, info

    def close(self):
        self.env.close()

    def set_task_idx(self, task_idx):
        self.env.set_task_idx(task_idx)
