import gym
import ipdb
import os
os.sys.path.append('..')

import carla_gym
import carla_gym.carla_wrapper as wrapper
# from config.carla import muzero_config
import imageio
import cv2
import numpy as np
# import ray
import time

import os
os.sys.path.append('/home/yihang/transfuser')
os.sys.path.append('/home/yihang/transfuser/team_code_recurrent')
os.sys.path.append('/home/yihang/transfuser/team_code_autopilot')
# import team_code_recurrent
from team_code_recurrent.submission_agent import HybridAgent, RoutePlanner
from team_code_autopilot.autopilot import AutoPilot


env = gym.make('LeaderBoard-v0',
               carla_map='Town01',
               host='localhost',
               port=1500,
               seed=0,
               no_rendering=False,
               weather_group='train',
               routes_group='train',
               routes_file_path='/home/yihang/transfuser/leaderboard/data/longest6/longest6_sep/longest6_Town01.xml',
               routes_file_format='standard',
               three_cam=True)
env = wrapper.CarlaWrapper(env, test=True)#, setting='pure_rgb')
env.set_task_idx(0)
obs_dict, info = env.reset()

apa = AutoPilot('/home/yihang/transfuser/log/54run031617/eval')

# hya = HybridAgent('/home/yihang/transfuser/log/54run031617/eval', info={'path':'path', 'name':'name'})
# hya._route_planner = RoutePlanner(7.5, 50.)
# hya._route_planner.set_route(info['global_plan'], coord=True)
# hya.initialized = True
# done = False
# step = 0
#
# while not done:
#     input_data = env.transfer_obs_to_transfuser_format(obs_dict)
#     ctrl = hya.run_step(input_data, 0)
#     step += 1
#     ac = {'steer': ctrl.steer, 'throttle': ctrl.throttle, 'brake': ctrl.brake}
#     obs_dict, rwd, done, info = env.step(ac)
#
#
# env.close()
