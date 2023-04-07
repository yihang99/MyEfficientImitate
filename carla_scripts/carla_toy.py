import time

import gym
import ipdb
import os
os.sys.path.append('..')
import numpy as np
import carla_gym
import carla_gym.carla_wrapper as wrapper

from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")
env = gym.make('LeaderBoard-v0',
               carla_map='Town01',
               host='localhost',
               port=1600,
               seed=0,
               no_rendering=False,
               weather_group='train',
               routes_group='train')
env = wrapper.CarlaWrapper(env, test=True, pure_rgb=True)
ipdb.set_trace()

model = SAC("MlpPolicy", env, verbose=1)
ipdb.set_trace()
print(time.ctime())
model.learn(total_timesteps=10000, log_interval=4)
print(time.ctime())# model.save("sac_pendulum")
#
# del model # remove to demonstrate saving and loading
#
# model = SAC.load("sac_pendulum")


obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     print(reward)
#     # env.render()
#     if done:
#       obs = env.reset()