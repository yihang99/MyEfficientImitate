import ipdb
from pathlib import Path

from carla_gym.carla_env import Carla_Env
import json
from carla_gym.utils.config_utils import parse_routes_file
from carla_gym import CARLA_GYM_ROOT_DIR
import yaml


class LeaderboardEnv(Carla_Env):
    def __init__(self, carla_map, host, port, seed, no_rendering, weather_group, routes_group=None, env_test=False,
                 three_cam=False):
        all_tasks = self.build_all_tasks(carla_map, weather_group, routes_group)
        print('len(all_tasks) =', len(all_tasks))
        self.three_cam = three_cam
        if self.three_cam:
            # obs_config_file = open('config/carla/obs_configs_three_cam.yaml')
            obs_config_file = open('../carla_config/carla/obs_configs_tf.yaml')
        else:
            obs_config_file = open('config/carla/obs_configs.yaml')
        obs_configs = yaml.load(obs_config_file, Loader=yaml.FullLoader)
        super().__init__(carla_map, host, port, obs_configs, all_tasks, seed, no_rendering, env_test)

    @staticmethod
    def build_all_tasks(carla_map, weather_group, routes_group=None):
        assert carla_map in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
        num_zombie_vehicles = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 70,
            'Town04': 150,
            'Town05': 120,
            'Town06': 120
        }
        num_zombie_walkers = {
            'Town01': 120,
            'Town02': 70,
            'Town03': 70,
            'Town04': 80,
            'Town05': 120,
            'Town06': 80
        }

        # weather
        if weather_group == 'new':
            weathers = ['SoftRainSunset', 'WetSunset']
        elif weather_group == 'train':
            weathers = ['ClearNoon', 'WetNoon', 'HardRainNoon', 'ClearSunset']
        elif weather_group == 'simple':
            weathers = ['ClearNoon']
        elif weather_group == 'train_eval':
            weathers = ['WetNoon', 'ClearSunset']
        elif weather_group == 'all':
            weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'SoftRainNoon', 'MidRainyNoon',
                        'HardRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
                        'SoftRainSunset', 'MidRainSunset', 'HardRainSunset']
        else:
            weathers = [weather_group]

        # task_type setup
        if carla_map == 'Town04' and routes_group is not None:
            description_folder = CARLA_GYM_ROOT_DIR / 'scenario_descriptions/LeaderBoard' \
                                 / f'Town04_{routes_group}'
        else:
            description_folder = CARLA_GYM_ROOT_DIR / 'scenario_descriptions/LeaderBoard' / carla_map

        actor_configs_dict = json.load(open(description_folder / 'actors.json'))
        route_descriptions_dict = parse_routes_file(description_folder / 'routes.xml')
        route_descriptions_dict = parse_routes_file(description_folder / 'routes_diy.xml')

        all_tasks = []
        for weather in weathers:
            for route_id, route_description in route_descriptions_dict.items():
                task = {
                    'weather': weather,
                    'description_folder': description_folder,
                    'route_id': route_id,
                    'num_zombie_vehicles': num_zombie_vehicles[carla_map],
                    'num_zombie_walkers': num_zombie_walkers[carla_map],
                    'ego_vehicles': {
                        'routes': route_description['ego_vehicles'],
                        'actors': actor_configs_dict['ego_vehicles'],
                    },
                    'scenario_actors': {
                        'routes': route_description['scenario_actors'],
                        'actors': actor_configs_dict['scenario_actors']
                    } if 'scenario_actors' in actor_configs_dict else {}
                }
                all_tasks.append(task)

        return all_tasks
