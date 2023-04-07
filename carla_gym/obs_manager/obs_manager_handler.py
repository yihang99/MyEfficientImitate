from importlib import import_module

import ipdb
from gym import spaces


class ObsManagerHandler(object):

    def __init__(self, obs_configs):
        self._obs_managers = {}
        self._obs_configs = obs_configs
        self._init_obs_managers()

    def get_observation(self):
        obs_dict = {}
        for om_id, om in self._obs_managers.items():
            obs_dict[om_id] = om.get_observation()
            # for obs_id, obs in om.get_observation().items():
            #     obs_dict[obs_id] = obs
        return obs_dict

    @property
    def observation_space(self):
        obs_spaces_dict = {}
        for om_id, om in self._obs_managers.items():
            obs_spaces_dict[om_id] = om.obs_space
            # for obs_id, obs_space in om.obs_space.spaces.items():
            #     obs_spaces_dict[obs_id] = obs_space
        return spaces.Dict(obs_spaces_dict)

    def reset(self, ego_vehicle):
        self._init_obs_managers()
        for obs_id, om in self._obs_managers.items():
            om.attach_ego_vehicle(ego_vehicle)

    def clean(self):
        for obs_id, om in self._obs_managers.items():
            om.clean()
        self._obs_managers = {}

    def _init_obs_managers(self):
        for obs_id, obs_config in self._obs_configs.items():
            ObsManager = getattr(import_module('carla_gym.obs_manager.' + obs_config["module"]), 'ObsManager')
            self._obs_managers[obs_id] = ObsManager(obs_config)
