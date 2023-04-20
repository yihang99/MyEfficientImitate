import time

import cv2
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

model = SAC("CnnPolicy", env, verbose=1)

start_time = time.time()
model.learn(total_timesteps=10000, log_interval=4)
print('training time:', time.time() - start_time)
model.save("carla_toy_ckpt")

# del model # remove to demonstrate saving and loading
model = SAC.load("carla_toy_ckpt")

obs = env.reset()
done = False
ipdb.set_trace()
step = 0
while not done:
    step += 1
    ac, _states = model.predict(obs, deterministic=True)
    ac = float(ac[0]), float(ac[1])
    obs, rwd, done, info = env.step(ac)
    im = obs  # obs['ori_data']
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('show', im)
    cv2.waitKey(20)
    print(step, '{:.6f}'.format(rwd), ac)

env.close()
