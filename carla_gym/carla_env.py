import carla
import gym
import time

import ipdb

from .ego_vehicle_handler import EgoVehicleHandler
from .obs_manager.obs_manager_handler import ObsManagerHandler
import numpy as np
from .util import set_random_seed
from .utils.traffic_light import TrafficLightHandler
#from .npc_agent import NpcAgent
from .zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .dynamic_weather import WeatherHandler


class Carla_Env(gym.Env):
    def __init__(self, carla_map, host, port, obs_configs, all_tasks, seed, no_rendering=True, env_test=False):
        super(Carla_Env, self).__init__()
        print("Begin to initialize the world")
        self._client = None
        self._world = None
        self._tm = None
        self._carla_map = carla_map
        self._all_tasks = all_tasks
        self._seed = seed
        #self.autopilot = autopilot

        self.monitor = None
        self._timestamp = None
        self._tl_handler = None
        #self._auto_agent = None
        #self.last_route_plan = None
        #self.last_spd = 0.
        #self.last_tl = None
        #self.last_st = None
        print('host:',host,'port:',port)

        self.init_client(carla_map, host, port, no_rendering)
        
        '''
        if self.autopilot:
            self._auto_agent = NpcAgent(resolution=1.0, target_speed=6.0,
                                        longitudinal_pid_params=[0.5, 0.025, 0.1],
                                        lateral_pid_params=[0.75, 0.05, 0.0],
                                        threshold_before=4.0,
                                        threshold_after=2.0)
        '''
        self._ev_handler = EgoVehicleHandler(self._client, self._tl_handler, test=env_test)
        self._om_handler = ObsManagerHandler(obs_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        # ToDo: refer the implementation of RouteScenario in Carla_env
        # self._sa_handler = ScenarioActorHandler(self._client)
        self._wt_handler = WeatherHandler(self._world)

        # observation spaces
        self.observation_space = self._om_handler.observation_space
        # define action spaces exposed to agent: throttle, steer, brake
        self.action_space = gym.spaces.Discrete(28)

        self._task_idx = 2
        self._shuffle_task = not True
        self._task = self._all_tasks[self._task_idx].copy()

    def reset(self):
        """
        :return: obs_dict, reward, done, info_dict
            obs_dict:
            info_dict:{
                'route_completion': {step, simulation time, route_completed_in_m, route_length_in_m, is_route_completed},
                'outside_route_lane': bool,
                'route_deviation': bool,
                'blocked': bool,
                'collision': bool,
                'run_red_light': bool,
                'encounter_light': bool,
                'run_stop_sign': bool
            }
        """
        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()
        self.clean()
        print('Carla_env._task_idx =', self._task_idx)

        ev_spawn_locations = self._ev_handler.reset(self._task['ego_vehicles'])
        self._zw_handler.reset(self._task['num_zombie_walkers'], ev_spawn_locations)
        self._zv_handler.reset(self._task['num_zombie_vehicles'], ev_spawn_locations)
        self._wt_handler.reset(self._task['weather'])
        self._om_handler.reset(self._ev_handler.ego_vehicle)
        self._world.tick()

        snap_shot = self._world.get_snapshot()
        self._timestamp = {
            'step': 0,
            'frame': snap_shot.timestamp.frame,
            'relative_wall_time': 0.0,
            'wall_time': snap_shot.timestamp.platform_timestamp,
            'relative_simulation_time': 0.0,
            'simulation_time': snap_shot.timestamp.elapsed_seconds,
            'start_frame': snap_shot.timestamp.frame,
            'start_wall_time': snap_shot.timestamp.platform_timestamp,
            'start_simulation_time': snap_shot.timestamp.elapsed_seconds
        }

        _, _, _ = self._ev_handler.tick(self.timestamp)
        obs_dict = self._om_handler.get_observation()
        '''
        self.last_route_plan = obs_dict['route_plan']
        self.last_spd = obs_dict['speed']['speed_xy'][0]
        self.last_st = info['run_stop_sign']
        self.last_tl = info['encounter_light']
        '''
        return obs_dict, {}

    def step(self, control_dict):
        #manual_stop_time = 189.5
        #new_control_dict = {}
        #print('autopilot',self.autopilot)
        '''
        if self.autopilot:
            expert_control = self._auto_agent.run_step(self.last_route_plan, self.last_spd, self.last_st, self.last_tl)
            new_control_dict['steer'] = expert_control.steer
            new_control_dict['throttle'] = expert_control.throttle
            new_control_dict['brake'] = expert_control.brake
            snap_shot = self._world.get_snapshot()
            
            if snap_shot.timestamp.elapsed_seconds > manual_stop_time:
                new_control_dict['steer'] = 0.0
                new_control_dict['throttle'] = 0.0
                new_control_dict['brake'] = 1.0
            
            self._ev_handler.apply_control(new_control_dict)
        else:
            self._ev_handler.apply_control(control_dict)
            '''
        self._ev_handler.apply_control(control_dict)
        # tick world
        self._world.tick()

        # update timestamp
        snap_shot = self._world.get_snapshot()
        self._timestamp['step'] = snap_shot.timestamp.frame - self._timestamp['start_frame']
        self._timestamp['frame'] = snap_shot.timestamp.frame
        self._timestamp['wall_time'] = snap_shot.timestamp.platform_timestamp
        self._timestamp['relative_wall_time'] = self._timestamp['wall_time'] - self._timestamp['start_wall_time']
        self._timestamp['simulation_time'] = snap_shot.timestamp.elapsed_seconds
        self._timestamp['relative_simulation_time'] = self._timestamp['simulation_time'] \
                                                      - self._timestamp['start_simulation_time']

        reward, done, info = self._ev_handler.tick(self.timestamp)

        obs_dict = self._om_handler.get_observation()
        '''
        if self.autopilot:
            obs_dict['supervision'] = new_control_dict
        '''
        # update weather
        # self._wt_handler.tick(snap_shot.timestamp.delta_seconds)

        # update the route_plan and speed
        '''
        self.last_route_plan = obs_dict['route_plan']
        self.last_spd = obs_dict['speed']['speed_xy'][0]
        self.last_st = info['run_stop_sign']
        self.last_tl = info['encounter_light']
        '''
        #if self.last_st:
            #print("Stop Sign:", self.last_st)

        # if self.last_tl:
        #     print('Traffic Light:', self.last_tl)

        return obs_dict, reward, done, info

    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()

    def init_client(self, carla_map, host, port, no_rendering=True):
        client = None
        while client is None:
            try:
                client = carla.Client(host, port)
                client.set_timeout(60.0)
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                client = None

        self._client = client
        while True:
            try:
                self._world = client.load_world(carla_map)
                break
            except RuntimeError as r:
                print(host, port, r)
                time.sleep(2)
                print('Trying Again')
        print('Success')
        self._tm = client.get_trafficmanager(port + 2)

        self.set_sync_mode(True)
        self.set_no_rendering_mode(self._world, no_rendering)

        set_random_seed(self._seed, using_cuda=True)
        self._tm.set_random_device_seed(self._seed)

        self._world.tick()

        # register traffic lights
        self._tl_handler = TrafficLightHandler()
        self._tl_handler.reset(self._world)

    def set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.1
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(sync)

    @staticmethod
    def set_no_rendering_mode(world, no_rendering):
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering
        world.apply_settings(settings)

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        print('Exiting the world ...')
        self.clean()
        self.set_sync_mode(False)
        self._client = None
        self._world = None
        self._tm = None

    def clean(self):
        # self._sa_handler.clean()
        # z clean ?
        self._zw_handler.clean()
        self._zv_handler.clean()
        self._om_handler.clean()
        self._ev_handler.clean()
        # self._wt_handler.clean()
        self._world.tick()

    def get_observation(self):
        pass

    def render(self, mode='human'):
        pass

    @property
    def num_tasks(self):
        return len(self._all_tasks)

    @property
    def task(self):
        return self._task

    @property
    def timestamp(self):
        return self._timestamp.copy()

    def get_traffic_lights_num(self):
        return self._tl_handler.num_tl
