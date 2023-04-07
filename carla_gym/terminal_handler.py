import numpy as np


class TerminalHandler(object):
    def __init__(self, ego_vehicle, no_collision=True, no_run_rl=True, no_run_stop=True, max_time=300):
        self._ego_vehicle = ego_vehicle

        self._no_collision = no_collision
        self._no_run_rl = no_run_rl
        self._no_run_stop = no_run_stop
        self._max_time = max_time  # in sec

        self._last_lat_dist = 0.0
        self._min_thresh_lat_dist = 3.5

    def get(self, timestamp):
        info_criteria = self._ego_vehicle.info_criteria

        # Done condition 1: blocked
        c_blocked = info_criteria['blocked'] is not None

        # Done condition 2: route_deviation
        c_route_deviation = info_criteria['route_deviation'] is not None

        # Done condition 3: collision
        c_collision = (info_criteria['collision'] is not None) and self._no_collision

        #ToDo: readd running red light
        # # Done condition 4: running red light
        # c_run_rl = (info_criteria['run_red_light'] is not None) and self._no_run_rl

        # Done condition 4: lateral distance too large
        ev_loc = self._ego_vehicle.vehicle.get_location()
        wp_transform = self._ego_vehicle.get_route_transform()
        d_vec = ev_loc - wp_transform.location
        np_d_vec = np.array([d_vec.x, d_vec.y], dtype=np.float32)
        wp_unit_forward = wp_transform.rotation.get_forward_vector()
        np_wp_unit_right = np.array([-wp_unit_forward.y, wp_unit_forward.x], dtype=np.float32)
        lat_dist = np.abs(np.dot(np_wp_unit_right, np_d_vec))

        if lat_dist - self._last_lat_dist > 0.8:
            thresh_lat_dist = lat_dist + 0.5
        else:
            thresh_lat_dist = max(self._min_thresh_lat_dist, self._last_lat_dist)
        c_lat_dist = lat_dist > thresh_lat_dist + 1e-2
        self._last_lat_dist = lat_dist

        # Done condition 5: lane_deviation
        c_outside_lane = info_criteria['outside_route_lane'] is not None

        # Done condition 5: run stop sign
        if info_criteria['run_stop_sign'] is not None and info_criteria['run_stop_sign']['event'] == 'run':
            c_run_stop = True
        else:
            c_run_stop = False
        c_run_stop = c_run_stop and self._no_run_stop

        # Done condition 6: timeout
        timeout = timestamp['relative_simulation_time'] > self._max_time

        # Done condition 7: route completed
        c_route = info_criteria['route_completion']['is_route_completed']

        done = c_blocked or c_route_deviation or c_collision or c_lat_dist or c_outside_lane or c_run_stop or timeout or c_route

        debug_texts = [
            f'deviation:{int(c_route_deviation)} blocked:{int(c_blocked)} timeout:{int(timeout)}',
            f'outside_lane:{int(c_outside_lane)} collision:{int(c_collision)}  run_st:{int(c_run_stop)}',
            f'finished:{int(c_route)}'
        ]

        terminal_debug = {
            'traffic_rule_violated': c_collision or c_run_stop,
            'blocked': c_blocked,
            'route_deviation': c_route_deviation,
            'debug_texts': debug_texts,
            'route_completed': c_route
        }

        terminal_reward = 0.
        if done:
            if not c_route:
                terminal_reward = -1.0
            else:
                terminal_reward = 1.0

        if c_lat_dist or c_collision or c_run_stop or c_collision:
            ev_vel = self._ego_vehicle.vehicle.get_velocity()
            ev_speed = np.linalg.norm(np.array([ev_vel.x, ev_vel.y]))
            terminal_reward -= ev_speed

        return done, timeout, terminal_reward, terminal_debug
