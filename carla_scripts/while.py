import gym
import ipdb
import os
os.sys.path.append('..')

import carla_gym
# from config.carla import muzero_config
import imageio
import cv2
import numpy as np
# import ray
import time

# for i in range(2):
#     im = np.random.rand(400,600,3)
#     cv2.imshow('noise_show', im)
#     cv2.waitKey(1000)
# print(time.ctime())
# cv2.waitKey(2000)
# print(time.ctime(), 'waited!')

if 1:
    # muzero_config.set_game('LeaderBoard-v0')
    # env = muzero_config.new_game(test=True, carla_map='Town01', host='localhost', port=1500, seed=0,
    #                              no_rendering=False, weather_group='train', routes_group='train', three_cam=True)
    env = gym.make('LeaderBoard-v0', carla_map='Town01', host='localhost', port=1500, seed=0, no_rendering=False, weather_group='train', routes_group='train', three_cam=True)

    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     ac = 22  # np.random.randint(0, 27)
    #     obs, rwd, done, info = env.step(ac)
    #     im = obs['ori_data']
    #     # im = cv2.resize(im, (480, 240))
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     cv2.imshow('show', im)
    #     # ns = np.random.rand(400, 600, 3)
    #     # cv2.imshow('noise_show', ns)
    #     cv2.waitKey(2)
    #     print(rwd, ac)
    env.close()
