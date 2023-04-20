import ipdb
import numpy as np
import carla

from .utils import transforms as trans_utils
from .object_finder.vehicle import ObsManager as OmVehicle
from .object_finder.pedestrian import ObsManager as OmPedestrian
from .utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker


class RewardHandler(object):
    def __init__(self, carla_map, ego_vehicle, tl_handler):
        self._map = carla_map
        self._ego_vehicle = ego_vehicle

        self.om_vehicle = OmVehicle({'max_detection_number': 10, 'distance_threshold': 20})
        self.om_pedestrian = OmPedestrian({'max_detection_number': 10, 'distance_threshold': 20})
        self.om_vehicle.attach_ego_vehicle(self._ego_vehicle)
        self.om_pedestrian.attach_ego_vehicle(self._ego_vehicle)

        self._maxium_speed = 6.0
        self._maxium_speed_junction = 5.0
        self._last_steer = 0.0
        self._tl_offset = -0.8 * self._ego_vehicle.vehicle.bounding_box.extent.x
        self._tl_handler = tl_handler

        self._just_cross_tl = False
        self._cross_tl_time = None

    def get(self, terminal_reward, simulation_time = None):
        ev_transform = self._ego_vehicle.vehicle.get_transform()
        ev_control = self._ego_vehicle.vehicle.get_control()
        ev_vel = self._ego_vehicle.vehicle.get_velocity()
        ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))

        # action
        if abs(ev_control.steer - self._last_steer) > 0.01:
            r_action = -0.2 - abs(ev_control.steer - self._last_steer)
        else:
            r_action = 0.0
        self._last_steer = ev_control.steer

        # desired_speed
        obs_vehicle = self.om_vehicle.get_observation()
        obs_pedestrian = self.om_pedestrian.get_observation()

        # all locations in ego_vehicle coordinate
        hazard_vehicle_loc = lbc_hazard_vehicle(obs_vehicle, proximity_threshold=9.5)
        hazard_ped_loc = lbc_hazard_walker(obs_pedestrian, proximity_threshold=9.5)
        light_state, light_loc, _ = self._tl_handler.get_light_state(self._ego_vehicle.vehicle,
                                                                     offset=self._tl_offset, dist_threshold=18.0)

        desired_spd_veh = desired_spd_ped = desired_spd_runLight = desired_spd_stop = desired_spd_junction = self._maxium_speed

        if hazard_vehicle_loc is not None:
            dist_veh = max(0.0, np.linalg.norm(hazard_vehicle_loc[0:2]) - 2.0)
            desired_spd_veh = self._maxium_speed * np.clip(dist_veh, 0.0, 11.0) / 11.0

        if hazard_ped_loc is not None:
            dist_ped = max(0.0, np.linalg.norm(hazard_ped_loc[0:2]) - 2.0)
            desired_spd_ped = self._maxium_speed * np.clip(dist_ped, 0.0, 9.0) / 9.0

        if light_state == carla.TrafficLightState.Red or light_state == carla.TrafficLightState.Yellow:
            dist_rl = max(0.0, np.linalg.norm(light_loc[0:2]) - 5.0)
            desired_spd_runLight = self._maxium_speed * np.clip(dist_rl, 0.0, 5.0) / 5.0
            self._just_cross_tl = True
            self._cross_tl_time  = simulation_time

        if light_state == carla.TrafficLightState.Green and self._just_cross_tl == True:
            desired_spd_runLight = self._maxium_speed * np.clip(simulation_time - self._cross_tl_time, 0.0, 5.0) / 5.0 + 0.1
            if simulation_time - self._cross_tl_time > 3:
                self._just_cross_tl = False
        # stop sign
        stop_sign = self._ego_vehicle.criteria_stop._target_stop_sign
        stop_loc = None
        if (stop_sign is not None) and (not self._ego_vehicle.criteria_stop._stop_completed):
            trans = stop_sign.get_transform()
            tv_loc = stop_sign.trigger_volume.location
            loc_in_world = trans.transform(tv_loc)
            loc_in_ev = trans_utils.loc_global_to_ref(loc_in_world, ev_transform)
            stop_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
            dist_stop = max(0.0, np.linalg.norm(stop_loc[0:2]) - 5.0)
            desired_spd_stop = self._maxium_speed * np.clip(dist_stop, 0.0, 5.0) / 5.0

        # junction
        current_loc = self._ego_vehicle.vehicle.get_location()
        current_waypoint = self._map.get_waypoint(current_loc, lane_type=carla.LaneType.Driving, project_to_road=True)
        if current_waypoint.is_junction:
            desired_spd_junction = self._maxium_speed_junction

        desired_speed = min(self._maxium_speed, desired_spd_veh, desired_spd_ped, desired_spd_runLight,
                            desired_spd_stop, desired_spd_junction)

        

        # r_position
        wp_transform = self._ego_vehicle.get_route_transform()
        d_vec = ev_transform.location - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)

        lateral_distance = np.abs(np.dot(np_wp_unit_right, np_d_vec))
        r_position = -1.0 * (lateral_distance / 2.0)

        # r_rotation
        angle_difference = np.abs(trans_utils.cast_angle(ev_transform.rotation.yaw - wp_transform.rotation.yaw))
        # r_rotation = -1.0 * (angle_difference / np.pi)
        r_rotation = -1.0 * np.deg2rad(angle_difference)
        # r_rotation = 1.0 - angle_difference / 180.0

        # r_speed

        if ev_speed > self._maxium_speed:
            # r_speed = 0.0
            r_speed = 1.0 - np.abs(ev_speed - desired_speed) / self._maxium_speed
        else:
            r_speed = 1.0 - np.abs(ev_speed - desired_speed) / self._maxium_speed

        if desired_speed < 0.1:
            r_speed = r_speed*np.cos(np.deg2rad(angle_difference))
        else:
            r_speed = 2*r_speed*np.cos(np.deg2rad(angle_difference))

        reward = r_speed + r_position + r_rotation + terminal_reward + r_action

        if hazard_vehicle_loc is None:
            txt_hazard_veh = '[]'
        else:
            txt_hazard_veh = np.array2string(hazard_vehicle_loc[0:2], precision=1, separator=',', suppress_small=True)
        if hazard_ped_loc is None:
            txt_hazard_ped = '[]'
        else:
            txt_hazard_ped = np.array2string(hazard_ped_loc[0:2], precision=1, separator=',', suppress_small=True)
        if light_loc is None:
            txt_light = '[]'
        else:
            txt_light = np.array2string(light_loc[0:2], precision=1, separator=',', suppress_small=True)
        if stop_loc is None:
            txt_stop = '[]'
        else:
            txt_stop = np.array2string(stop_loc[0:2], precision=1, separator=',', suppress_small=True)

        debug_texts = [
            f'r:{reward:5.2f} r_p:{r_position:5.2f} r_r:{r_rotation:5.2f}',
            f'ev_yaw:{ev_transform.rotation.yaw:5.2f}, wp_yaw:{wp_transform.rotation.yaw:5.2f}',
            f'ds:{desired_speed:5.2f} rs:{r_speed:5.2f} ra:{r_action:5.2f}',
            f'veh_ds:{desired_spd_veh:5.2f} {txt_hazard_veh}',
            f'ped_ds:{desired_spd_ped:5.2f} {txt_hazard_ped}',
            f'tl_ds:{desired_spd_runLight:5.2f} {light_state}{txt_light}',
            f'st_ds:{desired_spd_stop:5.2f} {txt_stop}',
            f'r_term:{terminal_reward:5.2f}'
        ]
        reward_debug = {
            'debug_texts': debug_texts,
            'r_speed': r_speed,
            'r_position': r_position,
            'r_rotation': r_rotation,
            'terminal_reward': terminal_reward,
            'r_action': r_action,
            'desired_spd_veh': desired_spd_veh,
            'desired_spd_ped': desired_spd_ped,
            'desired_spd_runLight': desired_spd_runLight,
            'desired_spd_stop': desired_spd_stop, 
            'desired_spd_junction': desired_spd_junction,
            'simulation_time': simulation_time

        }
        # ipdb.set_trace()
        return reward, reward_debug
