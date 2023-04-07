import carla
import time
import numpy as np
import random
import math
from .task_vehicle import TaskVehicle

from .reward_handler import RewardHandler
from .terminal_handler import TerminalHandler

PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80


class EgoVehicleHandler(object):
    def __init__(self, client, tl_handler, test=False):
        self.ego_vehicle = None
        self._world = client.get_world()
        self._map = self._world.get_map()
        self._spawn_transforms = self._get_spawn_points(self._map)
        self.reward_handler = None
        self.terminal_handler = None
        self._tl_handler = tl_handler
        self.test = test

        self.reward_buffers = []
        self.info_buffers = {
            'collisions_layout': [],
            'collisions_vehicle': [],
            'collisions_pedestrian': [],
            'collisions_others': [],
            'red_light': [],
            'encounter_light': [],
            'stop_infraction': [],
            'encounter_stop': [],
            'route_dev': [],
            'vehicle_blocked': [],
            'outside_lane': [],
            'wrong_lane': []
        }

    def reset(self, task_config):
        actor_config = task_config['actors']
        route_config = task_config['routes']
        vehicle_bp = self._world.get_blueprint_library().find(actor_config['hero']['model'])
        vehicle_bp.set_attribute('role_name', 'hero')
        spawn_transform = route_config['hero'][0]

        # Give the z-direction a small value or it will collides with the street
        wp = self._map.get_waypoint(spawn_transform.location)
        spawn_transform.location.z = wp.transform.location.z + 1.5
        carla_vehicle = self._world.try_spawn_actor(vehicle_bp, spawn_transform)
        self._world.tick()
        time.sleep(2.0)
        # the vehicle needs around 25 steps to get ready
        for _ in range(30):
            control = carla.VehicleControl()
            control.steer = random.uniform(-0.1, 0.1)
            control.throttle = 0.2
            control.brake = 0.0
            carla_vehicle.apply_control(control)
            self._world.tick()

        vel = carla_vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        while speed > 0:
            carla_vehicle.apply_control(carla.VehicleControl(throttle=0., steer=0., brake=1.))
            vel = carla_vehicle.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            self._world.tick()

        target_transforms = route_config['hero'][1:]
        self.ego_vehicle = TaskVehicle(carla_vehicle, target_transforms, self._spawn_transforms, self._tl_handler,
                                       test=self.test)
        self.reward_handler = RewardHandler(self._map, self.ego_vehicle, self._tl_handler)
        self.terminal_handler = TerminalHandler(self.ego_vehicle)

        return [carla_vehicle.get_location()]

    def tick(self, timestamp):
        info_criteria = self.ego_vehicle.tick(timestamp)
        info = info_criteria.copy()
        done, timeout, terminal_reward, terminal_debug = self.terminal_handler.get(timestamp)
        reward, reward_debug = self.reward_handler.get(terminal_reward, timestamp['simulation_time'])

        info['timeout'] = timeout
        info['reward_debug'] = reward_debug
        info['terminal_debug'] = terminal_debug
        # accumulate into reward buffers
        self.reward_buffers.append(reward)

        # update details to info_buffers
        if info['collision']:
            if info['collision']['collision_type'] == 0:
                self.info_buffers['collisions_layout'].append(info['collision'])
            elif info['collision']['collision_type'] == 1:
                self.info_buffers['collisions_vehicle'].append(info['collision'])
            elif info['collision']['collision_type'] == 2:
                self.info_buffers['collisions_pedestrian'].append(info['collision'])
            else:
                self.info_buffers['collisions_others'].append(info['collision'])

        if info['run_red_light']:
            self.info_buffers['red_light'].append(info['run_red_light'])

        if info['encounter_light']:
            self.info_buffers['encounter_light'].append(info['encounter_light'])

        if info['run_stop_sign']:
            if info['run_stop_sign']['event'] == 'encounter':
                self.info_buffers['encounter_stop'].append(info['run_stop_sign'])
            elif info['run_stop_sign']['event'] == 'run':
                self.info_buffers['stop_infraction'].append(info['run_stop_sign'])

        if info['route_deviation']:
            self.info_buffers['route_dev'].append(info['route_deviation'])

        if info['blocked']:
            self.info_buffers['vehicle_blocked'].append(info['blocked'])

        if info['outside_route_lane']:
            if info['outside_route_lane']['outside_lane']:
                self.info_buffers['outside_lane'].append(info['outside_route_lane'])
            if info['outside_route_lane']['wrong_lane']:
                self.info_buffers['wrong_lane'].append(info['outside_route_lane'])

        # save episode summary
        if done:
            info['episode_event'] = self.info_buffers
            info['episode_event']['timeout'] = info['timeout']
            info['episode_event']['route_completion'] = info['route_completion']

            total_length = float(info['route_completion']['route_length_in_m']) / 1000
            completed_length = float(info['route_completion']['route_completed_in_m']) / 1000
            total_length = max(total_length, 0.001)
            completed_length = max(completed_length, 0.001)
            if info['route_completion']['is_route_completed']:
                score_route = 1.0
            else:
                score_route = completed_length / total_length

            outside_lane_length = np.sum([x['distance_traveled']
                                          for x in self.info_buffers['outside_lane']]) / 1000
            wrong_lane_length = np.sum([x['distance_traveled']
                                        for x in self.info_buffers['wrong_lane']]) / 1000

            n_collisions_layout = int(len(self.info_buffers['collisions_layout']))
            n_collisions_vehicle = int(len(self.info_buffers['collisions_vehicle']))
            n_collisions_pedestrian = int(len(self.info_buffers['collisions_pedestrian']))
            n_collisions_others = int(len(self.info_buffers['collisions_others']))
            n_red_light = int(len(self.info_buffers['red_light']))
            n_encounter_light = int(len(self.info_buffers['encounter_light']))
            n_stop_infraction = int(len(self.info_buffers['stop_infraction']))
            n_encounter_stop = int(len(self.info_buffers['encounter_stop']))
            n_collisions = n_collisions_layout + n_collisions_vehicle + n_collisions_pedestrian + n_collisions_others

            score_penalty = 1.0 * (1 - (outside_lane_length + wrong_lane_length) / completed_length) \
                            * (PENALTY_COLLISION_STATIC ** n_collisions_layout) \
                            * (PENALTY_COLLISION_VEHICLE ** n_collisions_vehicle) \
                            * (PENALTY_COLLISION_PEDESTRIAN ** n_collisions_pedestrian) \
                            * (PENALTY_TRAFFIC_LIGHT ** n_red_light) \
                            * (PENALTY_STOP ** n_stop_infraction)

            if info['route_completion']['is_route_completed'] and n_collisions == 0:
                is_route_completed_nocrash = 1.0
            else:
                is_route_completed_nocrash = 0.0

            info['episode_stat'] = {
                'score_route': score_route,
                'score_penalty': score_penalty,
                'score_composed': max(score_route * score_penalty, 0.0),
                'length': len(self.reward_buffers),
                'reward': np.sum(self.reward_buffers),
                'timeout': float(info['timeout']),
                'is_route_completed': float(info['route_completion']['is_route_completed']),
                'is_route_completed_nocrash': is_route_completed_nocrash,
                'route_completed_in_km': completed_length,
                'route_length_in_km': total_length,
                'percentage_outside_lane': outside_lane_length / completed_length,
                'percentage_wrong_lane': wrong_lane_length / completed_length,
                'collisions_layout': n_collisions_layout / completed_length,
                'collisions_vehicle': n_collisions_vehicle / completed_length,
                'collisions_pedestrian': n_collisions_pedestrian / completed_length,
                'collisions_others': n_collisions_others / completed_length,
                'red_light': n_red_light / completed_length,
                'light_passed': n_encounter_light - n_red_light,
                'encounter_light': n_encounter_light,
                'stop_infraction': n_stop_infraction / completed_length,
                'stop_passed': n_encounter_stop - n_stop_infraction,
                'encounter_stop': n_encounter_stop,
                'route_deviation': len(self.info_buffers['route_dev']) / completed_length,
                'vehicle_blocked': len(self.info_buffers['vehicle_blocked']) / completed_length
            }

        return reward, done, info

    def apply_control(self, control_dict):
        control = carla.VehicleControl()
        control.steer = control_dict['steer']
        control.throttle = control_dict['throttle']
        control.brake = control_dict['brake']
        self.ego_vehicle.vehicle.apply_control(control)

    @staticmethod
    def _get_spawn_points(carla_map):
        all_spawn_points = carla_map.get_spawn_points()
        spawn_transforms = []
        for trans in all_spawn_points:
            wp = carla_map.get_waypoint(trans.location)

            if wp.is_junction:
                wp_prev = wp
                # wp_next = wp
                while wp_prev.is_junction:
                    wp_prev = wp_prev.previous(1.0)[0]
                spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                if carla_map.name == 'Town03' and (wp_prev.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                # while wp_next.is_junction:
                #     wp_next = wp_next.next(1.0)[0]

                # spawn_transforms.append([wp_next.road_id, wp_next.transform])
                # if c_map.name == 'Town03' and (wp_next.road_id == 44 or wp_next.road_id == 58):
                #     for _ in range(100):
                #         spawn_transforms.append([wp_next.road_id, wp_next.transform])

            else:
                spawn_transforms.append([wp.road_id, wp.transform])
                if carla_map.name == 'Town03' and (wp.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp.road_id, wp.transform])

        return spawn_transforms

    def clean(self):
        if self.ego_vehicle:
            self.ego_vehicle.clean()
        self.reward_handler = None
        self.terminal_handler = None
        self.reward_buffers = []
        self.info_buffers = {
            'collisions_layout': [],
            'collisions_vehicle': [],
            'collisions_pedestrian': [],
            'collisions_others': [],
            'red_light': [],
            'encounter_light': [],
            'stop_infraction': [],
            'encounter_stop': [],
            'route_dev': [],
            'vehicle_blocked': [],
            'outside_lane': [],
            'wrong_lane': []
        }
