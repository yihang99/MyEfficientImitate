from collections import deque
import carla
import numpy as np
import carla_gym.utils.transforms as trans_utils


def _get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    adapted from "carla-simulator/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py"
    """
    base_transform = traffic_light.get_transform()
    tv_loc = traffic_light.trigger_volume.location
    tv_ext = traffic_light.trigger_volume.extent

    # Discretize the trigger box into points
    x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
    area = []
    for x in x_values:
        point_location = base_transform.transform(tv_loc + carla.Location(x=x))
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Leaderboard: Advance them until the intersection
    stopline_wps = []
    stopline_vertices = []
    junction_wps = []
    for wpx in ini_wps:
        # Below: just use trigger volume, otherwise it's on the zebra lines.
        # stopline_wps.append(wpx)
        # vec_forward = wpx.transform.get_forward_vector()
        # vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        # loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        # loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        # stopline_vertices.append([loc_left, loc_right])

        # get the waypoints of intersection area
        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        junction_wps.append(wpx)

        stopline_wps.append(wpx)
        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    # all paths at junction for this traffic light
    junction_paths = []
    path_wps = []
    wp_queue = deque(junction_wps)
    while len(wp_queue) > 0:
        current_wp = wp_queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                wp_queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths


class TrafficLightHandler(object):
    num_tl = 0
    list_tl_actor = []
    list_tv_loc = []
    list_stopline_wps = []
    list_stopline_vtx = []
    list_junction_paths = []
    carla_map = None

    def reset(self, world):
        self.carla_map = world.get_map()

        self.num_tl = 0
        self.list_tl_actor = []
        self.list_tv_loc = []
        self.list_stopline_wps = []
        self.list_stopline_vtx = []
        self.list_junction_paths = []

        all_actors = world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints(
                    _actor, self.carla_map)

                self.list_tl_actor.append(_actor)
                self.list_tv_loc.append(tv_loc)
                self.list_stopline_wps.append(stopline_wps)
                self.list_stopline_vtx.append(stopline_vtx)
                self.list_junction_paths.append(junction_paths)

                self.num_tl += 1

    def get_light_state(self, vehicle, offset=0.0, dist_threshold=15.0):
        """
        vehicle: carla.Vehicle
        """
        vec_tra = vehicle.get_transform()
        veh_dir = vec_tra.get_forward_vector()

        hit_loc = vec_tra.transform(carla.Location(x=offset))
        hit_wp = self.carla_map.get_waypoint(hit_loc)

        light_loc = None
        light_state = None
        light_id = None
        for i in range(self.num_tl):
            traffic_light = self.list_tl_actor[i]
            tv_loc = 0.5*self.list_stopline_wps[i][0].transform.location \
                + 0.5*self.list_stopline_wps[i][-1].transform.location

            distance = np.sqrt((tv_loc.x-hit_loc.x)**2 + (tv_loc.y-hit_loc.y)**2)
            if distance > dist_threshold:
                continue

            for wp in self.list_stopline_wps[i]:

                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = veh_dir.x * wp_dir.x + veh_dir.y * wp_dir.y + veh_dir.z * wp_dir.z

                wp_1 = wp.previous(4.0)[0]
                same_road = (hit_wp.road_id == wp.road_id) and (hit_wp.lane_id == wp.lane_id)
                same_road_1 = (hit_wp.road_id == wp_1.road_id) and (hit_wp.lane_id == wp_1.lane_id)

                # if (wp.road_id != wp_1.road_id) or (wp.lane_id != wp_1.lane_id):
                #     print(f'Traffic Light Problem: {wp.road_id}={wp_1.road_id}, {wp.lane_id}={wp_1.lane_id}')

                if (same_road or same_road_1) and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    loc_in_ev = trans_utils.loc_global_to_ref(wp.transform.location, vec_tra)
                    light_loc = np.array([loc_in_ev.x, loc_in_ev.y, loc_in_ev.z], dtype=np.float32)
                    light_state = traffic_light.state
                    light_id = traffic_light.id
                    break

        return light_state, light_loc, light_id

    def get_junctoin_paths(self, veh_loc, color=0, dist_threshold=50.0):
        if color == 0:
            tl_state = carla.TrafficLightState.Green
        elif color == 1:
            tl_state = carla.TrafficLightState.Yellow
        elif color == 2:
            tl_state = carla.TrafficLightState.Red

        junctoin_paths = []
        for i in range(self.num_tl):
            traffic_light = self.list_tl_actor[i]
            tv_loc = self.list_tv_loc[i]
            if tv_loc.distance(veh_loc) > dist_threshold:
                continue
            if traffic_light.state != tl_state:
                continue

            junctoin_paths += self.list_junction_paths[i]

        return junctoin_paths

    def get_stopline_vtx(self, veh_loc, color, dist_threshold=50.0):
        if color == 0:
            tl_state = carla.TrafficLightState.Green
        elif color == 1:
            tl_state = carla.TrafficLightState.Yellow
        elif color == 2:
            tl_state = carla.TrafficLightState.Red

        stopline_vtx = []
        for i in range(self.num_tl):
            traffic_light = self.list_tl_actor[i]
            tv_loc = self.list_tv_loc[i]
            if tv_loc.distance(veh_loc) > dist_threshold:
                continue
            if traffic_light.state != tl_state:
                continue
            stopline_vtx += self.list_stopline_vtx[i]

        return stopline_vtx
